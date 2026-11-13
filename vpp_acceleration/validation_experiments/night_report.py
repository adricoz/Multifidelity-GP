"""
Report of the validation night of 2026-11-12 (French, for Adrian and Antoine): VALIDATION_2026-11-13.md with
the pass / fail table of the plan, from whatever results exist when it is called (night.py calls it
after every step), and the figures of both analysed runs (figures.py). Never fails on a missing part.

    ..\\.venv-mds\\Scripts\\python validation_experiments\\night_report.py
"""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TWIN = HERE.parent / "twin_experiments"
R11, RNO = HERE / "results", HERE / "results" / "nocfd"
TOL = 0.02                                   # kt, criterion fixed before the night
CV_LIMIT = 1.3

PURPOSE = [
    "## À quoi sert chaque test, et ce qu'il montrera", "",
    "La question : la section trouvée (+0,39 nd de vitesse moyenne sur la section actuelle) est-elle "
    "le meilleur choix du domaine, à 0,02 nd près, ou seulement le meilleur de ce que la méthode a "
    "regardé ? Il n'y a pas de descente de gradient sur les vrais codes, et le co-krigeage est "
    "entraîné sur des points **ciblés** autour de quelques sections : il pourrait être juste là où il "
    "a des données et se tromper ailleurs sans le savoir. L'affirmation « pas de meilleur optimum "
    "ailleurs » repose sur une chaîne de maillons ; chaque test en vérifie un.", "",
    "| Maillon | Test |", "|---|---|",
    "| Le modèle est juste là où il a des données | déjà fait : validation croisée classique, "
    "validation de fin de run par les codes |",
    "| Son incertitude est honnête **loin** des données | 1 |",
    "| Avec cette incertitude, aucune région ne peut cacher une meilleure section | 2 |",
    "| Le résultat ne dépend pas de l'histoire du run (ciblage, graine, données) | 3 et 4 |",
    "| L'algorithme retrouve un optimum connu | 5 |",
    "| Le modèle ne se trompe pas sur des sections éloignées, d'après les codes | 6 (non lancé) |",
    "| Le gain ne dépend pas du profil d'usage choisi | 7 |", "",
    "**1. Validation croisée avec zone d'exclusion.** Pour un point test, on retire *tous* les points "
    "dont la section est à moins de r scores (distance dans l'espace des 10 modes ACP), on le prédit "
    "avec ce qui reste, et on divise l'erreur par l'écart-type annoncé (« écart réduit »). Rayons 0 ; "
    "0,25 ; 0,5 ; 1 ; 2 ; 3 ; 300 points tests par niveau ; les 6 efforts. À r = 0 c'est la validation "
    "croisée habituelle, optimiste parce qu'un point voisin presque identique reste dans les données ; "
    "à r ≥ 0,5 on se met dans la situation du criblage : juger une section qu'on n'a jamais calculée. "
    "*Ce que ça montre* : un écart réduit proche de 1 (ou en dessous) veut dire que, loin des données, "
    "le modèle sait qu'il ne sait pas : quand il annonce « rien de mieux ici » avec une petite "
    "incertitude, on peut le croire. Au-dessus de 1, il est trop sûr de lui, et le facteur mesuré à "
    "1 score (κ_loin) sert à gonfler l'incertitude du certificat. Critère : ≤ 1,3 à 1 score au "
    "niveau 2.", "",
    "**2. Certificat (borne haute maximale).** On cherche, sur tout le domaine admissible (mêmes "
    "contraintes que le run, Ixx/v compris), la section qui maximise *gain prédit + κ × écart-type du "
    "gain* par rapport à la meilleure section, avec de nombreux départs (sections du run et optima "
    "locaux). Le gain est la vitesse moyenne pondérée par le profil, au premier ordre autour des états "
    "VPP de la meilleure section, avec la covariance exacte du modèle. Fait pour κ = 1, 2, 3 et κ_loin, "
    "au niveau AVL × XFOIL et au niveau CFD du 11/11, et au niveau AVL × XFOIL du run sans CFD. "
    "*Ce que ça montre* : si même cette borne optimiste reste sous +0,02 nd, alors **nulle part** dans "
    "le domaine le modèle ne laisse de place à une section nettement meilleure, y compris dans les "
    "régions qu'il connaît mal (là où σ est grand, la borne monte d'elle-même). C'est la réponse "
    "directe à « comment savoir qu'il n'y a pas d'optimum ailleurs ». Si la borne dépasse, la figure "
    "dit **où** (distance à la meilleure section) : c'est là qu'il faudrait calculer. Les sections qui "
    "réalisent la borne sont ensuite résolues par la vraie VPP sur la vérité du jumeau, ce qui enlève "
    "l'approximation au premier ordre.", "",
    "**3. Jugement croisé des deux runs.** Le run du 11/11 (3 niveaux, CFD) et le run sans CFD "
    "(2 niveaux) ont des données, des graines et des ciblages différents, dans le même espace de "
    "sections. Chaque modèle évalue la meilleure section de l'autre (écart et incertitude), puis la "
    "VPP exacte évalue la section du run sans CFD sur le modèle du 11/11. *Ce que ça montre* : si "
    "chaque modèle place l'optimum de l'autre à moins de 0,02 nd du sien, le ciblage n'a pas décidé "
    "du résultat ; deux apprentissages différents s'accordent. Si un modèle juge l'optimum de l'autre "
    "nettement moins bon alors que l'autre le juge meilleur, c'est le signe d'un biais de ciblage.", "",
    "**4. Tests existants rejoués sur le run sans CFD.** Test KKT (le gradient de la vitesse est-il "
    "nul, aux contraintes actives près ?), montée de gradient depuis la meilleure section, regret "
    "attendu. *Ce que ça montre* : que les conclusions « optimum local sur le modèle » du 11/11 se "
    "reproduisent sur un second run, et ne sont pas un accident d'un seul jeu de données.", "",
    "**5. Jumeau numérique.** On remplace les codes aéro par une fonction **connue** et gratuite : la "
    "moyenne du modèle du 11/11 (niveau 1 et niveau 2). Sur elle, on cherche le vrai optimum "
    "*directement* : montée à région de confiance, chaque pas vérifié par la vraie VPP, depuis "
    "plusieurs départs (meilleure section, section actuelle, optima locaux du run, une section au "
    "hasard). En parallèle, la méthode complète repart **de zéro** sur cette vérité (plan initial, "
    "ciblage, criblage, redémarrages, convergence, mêmes options que le run sans CFD, autre graine). "
    "On compare les vitesses vraies. *Ce que ça montre* : c'est le seul test où l'on connaît la "
    "réponse. Perte ≤ 0,02 nd : l'algorithme retrouve l'optimum. Perte plus grande avec une section "
    "éloignée : il s'est arrêté dans un autre bassin (criblage ou redémarrages insuffisants). Section "
    "éloignée mais même vitesse : la vérité a une crête plate, l'optimum n'est pas unique et ce n'est "
    "pas grave. Si les départs de la recherche directe arrivent à des vitesses différentes, la vérité "
    "a plusieurs bassins et le tableau du jumeau dit lesquels comptent. Limite : la vérité est un "
    "krigeage, plus lisse que les vrais codes ; le test juge l'algorithme, pas la régularité des codes "
    "(c'est le rôle des tests 1 et 6).", "",
    "**6. Verdict des codes (non lancé aujourd'hui).** AVL × XFOIL réels aux sections du certificat. "
    "Il montrerait si le modèle se trompe sur des sections éloignées d'après les vrais codes. À "
    "décider ensemble.", "",
    "**7. Profil d'usage et direction active.** Le gain est recalculé à partir des 49 vitesses de la "
    "validation, en multipliant le poids de chaque mode (près, portant, vent arrière, manuel) de 0 à "
    "3 fois, les autres renormalisés ; plus la carte du gain par condition et la vitesse prédite de "
    "toutes les sections calculées le long de la direction qui porte la vitesse. *Ce que ça montre* : "
    "si le gain garde son signe quel que soit le profil, le choix de section ne dépend pas d'un "
    "profil d'usage qu'on a fixé à la main ; la carte montre où le gain se fait (et où l'on perd) ; "
    "la coupe montre une bosse ou plusieurs.", "",
    "**Ensemble.** Si 1, 2, 3, 4 et 5 réussissent : le modèle est honnête loin des données, il ne "
    "laisse de place nulle part à une section meilleure de plus de 0,02 nd, deux runs indépendants "
    "s'accordent, et l'algorithme retrouve un optimum connu. Cela ne prouve pas que les codes (ni la "
    "CFD) sont d'accord avec le modèle loin des données : c'est le test 6, puis une vérification CFD "
    "de la meilleure section contre la section actuelle.", ""]


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def fr(value, digits: int = 3, sign: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:{'+' if sign else ''}.{digits}f}".replace(".", ",")


def verdict(ok) -> str:
    return "pas de résultat" if ok is None else "**réussi**" if ok else "**échoué**"


def kappa_far(folder: Path, level: str = "2", radius: str = "1.0"):
    data = read(folder / "buffered_cv.json")
    return None if data is None else data["kappa_far"].get(level, {}).get(radius)


def bound(folder: Path, level: int, kappa: float):
    data = read(folder / f"certificate_level{level}.json")
    if data is None or kappa is None:
        return None
    gain, sigma = np.array(data["all"]["gain"]), np.array(data["all"]["sigma"])
    return float((gain + kappa * sigma).max())


def cv_rows(lines: list) -> None:
    for label, folder in (("run du 11/11", R11), ("run sans CFD", RNO)):
        data = read(folder / "buffered_cv.json")
        if data is None:
            lines.append(f"| 1 | Validation croisée avec zone d'exclusion, {label} | écart réduit ≤ "
                         f"{fr(CV_LIMIT, 1)} à 1 score | – | pas de résultat |")
            continue
        values = {level: data["kappa_far"].get(level, {}) for level in ("1", "2")}
        text = " ; ".join(f"niveau {level} : " + ", ".join(
            f"{fr(v, 2)} à r = {float(r):g}" for r, v in values[level].items() if float(r) in (0, 1, 2))
            for level in ("1", "2") if values[level])
        at_one = values["2"].get("1.0")
        lines.append(f"| 1 | Validation croisée avec zone d'exclusion, {label} | écart réduit ≤ "
                     f"{fr(CV_LIMIT, 1)} à 1 score (niveau 2) | {text} | "
                     f"{verdict(None if at_one is None else at_one <= CV_LIMIT)} |")


def certificate_rows(lines: list) -> None:
    for label, folder, level in (("run du 11/11, niveau 2", R11, 2), ("run du 11/11, niveau CFD", R11, 3),
                                 ("run sans CFD", RNO, 2)):
        far = kappa_far(folder)
        values = [bound(folder, level, k) for k in (1.0, 2.0)]
        with_far = bound(folder, level, max(far, 1.0)) if far is not None else None
        if values[0] is None:
            lines.append(f"| 2 | Certificat, {label} | borne haute ≤ {fr(TOL, 2)} nd | – | pas de résultat |")
            continue
        text = (f"κ = 1 : {fr(values[0], 3, True)} nd ; κ = 2 : {fr(values[1], 3, True)} nd"
                + (f" ; κ_loin = {fr(max(far, 1.0), 2)} : {fr(with_far, 3, True)} nd"
                   if with_far is not None else ""))
        judged = with_far if with_far is not None else values[0]
        lines.append(f"| 2 | Certificat, {label} | borne haute ≤ {fr(TOL, 2)} nd (avec κ_loin) | {text} | "
                     f"{verdict(judged <= TOL)} |")


def cross_rows(lines: list, compare) -> None:
    data = read(R11 / "cross.json")
    if data is None or not data.get("judgements"):
        lines.append("| 3 | Jugement croisé des deux runs (premier ordre) | écart ≤ 0,02 nd | – | "
                     "pas de résultat |")
    else:
        parts, ok = [], True
        for judgement in data["judgements"]:
            if judgement["level"] != 2:
                continue
            i = judgement["names"].index("other_best")
            gap, sigma = judgement["gain_kt"][i], judgement["sigma_kt"][i]
            parts.append(f"juge {'11/11' if 'cfd3d' in judgement['judge'] else 'sans CFD'} : "
                         f"{fr(gap, 3, True)} ± {fr(sigma, 3)} nd")
            ok = ok and abs(gap) <= TOL
        lines.append("| 3 | Jugement croisé (premier ordre, niveau 2) : l'optimum de l'autre run vu par "
                     f"chaque modèle | écart ≤ 0,02 nd | {' ; '.join(parts)} (distance entre optima "
                     f"{fr(data['distance_between_best'], 2)} score) | {verdict(ok if parts else None)} |")
    if compare is not None:
        rows = {r["name"]: r for r in compare["sections"]}
        if "nocfd_best" in rows and rows["nocfd_best"]["gap_to_anchor_kt"] is not None:
            gap = rows["nocfd_best"]["gap_to_anchor_kt"]
            lines.append("| 3 | Jugement exact (VPP) : optimum du run sans CFD sur la vérité du jumeau "
                         "(modèle du 11/11, niveau 2) | à moins de 0,02 nd du meilleur optimum de "
                         f"référence | {fr(gap, 4, True)} nd, à {fr(rows['nocfd_best']['distance_to_anchor'], 2)} "
                         f"score | {verdict(gap >= -TOL)} |")


def rerun_rows(lines: list) -> None:
    gradient, regret = read(RNO / "gradient.json"), read(RNO / "regret.json")
    if gradient is None:
        lines.append("| 4 | Tests existants rejoués sur le run sans CFD | KKT satisfaites, montée de "
                     "gradient < 0,01 nd | – | pas de résultat |")
        return
    p_value = gradient["kkt"]["1"]["p_value"]
    ascent = max(a["gain_kt"] for a in gradient["ascent"])
    text = (f"KKT : p = {fr(p_value, 2)} ; montée de gradient au plus {fr(ascent, 4, True)} nd")
    if regret is not None:
        text += (f" ; regret attendu {fr(regret['scales']['1.0']['expected_regret_kt'], 3)} nd, "
                 f"P(mieux de > 0,02 nd) = {fr(100 * regret['scales']['1.0']['p_better_by']['0.02'], 0)} %")
    lines.append("| 4 | Tests existants rejoués sur le run sans CFD | KKT non rejetées (p > 0,05), "
                 f"montée de gradient < 0,01 nd | {text} | {verdict(p_value > 0.05 and ascent < 0.01)} |")


def twin_row(lines: list, compare) -> None:
    result = None if compare is None else compare.get("verdict")
    if result is None:
        lines.append("| 5 | Jumeau numérique : la méthode retrouve-t-elle l'optimum connu ? | perte ≤ "
                     "0,02 nd | – | pas de résultat |")
        return
    lines.append("| 5 | Jumeau numérique : la méthode retrouve-t-elle l'optimum connu ? | perte ≤ 0,02 nd "
                 f"| perte {fr(result['regret_kt'], 4, True)} nd, à {fr(result['distance_scores'], 2)} score "
                 f"du meilleur optimum de la recherche directe | {verdict(result['passed'])} |")


def certificate_on_truth(lines: list, compare) -> None:
    if compare is None:
        return
    rows = [r for r in compare["sections"] if r["group"] == "certificate"
            and r["gap_to_anchor_kt"] is not None]
    if rows:
        best = max(rows, key=lambda r: r["gap_to_anchor_kt"])
        lines.append("| 2 | Sections de borne haute maximale, résolues par la VPP sur la vérité du jumeau | "
                     "aucune ne bat le meilleur optimum de référence de plus de 0,02 nd | meilleure : "
                     f"{fr(best['gap_to_anchor_kt'], 4, True)} nd ({len(rows)} sections) | "
                     f"{verdict(best['gap_to_anchor_kt'] <= TOL)} |")


def steps_table(status) -> list:
    lines = ["| Étape | État | Début | Durée (min) |", "|---|---|---|---|"]
    names = {"done": "terminée", "failed": "échec", "killed": "arrêtée à l'heure limite",
             "skipped": "non lancée", "running": "en cours"}
    for name, entry in status["steps"].items():
        start = str(entry.get("start", ""))[11:16]
        minutes = "–"
        if entry.get("end") and entry.get("start"):
            try:
                span = (dt.datetime.fromisoformat(str(entry["end"]))
                        - dt.datetime.fromisoformat(str(entry["start"])))
                minutes = f"{span.total_seconds() / 60:.0f}"
            except ValueError:
                pass
        lines.append(f"| `{name}` | {names.get(entry['state'], entry['state'])} | {start} | {minutes} |")
    return lines


def main():
    for results, figures in (("results", "figures"), ("results/nocfd", "figures/nocfd")):
        if (HERE / results).exists():
            subprocess.run([sys.executable, str(HERE / "figures.py"), "--results", results,
                            "--figures", figures], cwd=str(HERE), check=False, timeout=600)
    status = read(HERE / "logs" / "night" / "status.json") or {"steps": {}, "notes": []}
    compare = read(TWIN / "results" / "compare.json")
    finished = any("night finished" in note for note in status["notes"])
    lines = ["# 13/11/2026 : validation de l'algorithme d'optimisation (sans CFD)", "",
             f"Compte rendu écrit automatiquement le {dt.datetime.now():%d/%m/%Y à %H:%M} par "
             f"`night_report.py` — série de tests **{'terminée' if finished else 'en cours'}**. Plan approuvé le "
             "12/11, lancé le 13/11 ; critères fixés avant de lancer.", "",
             "## Le tableau des tests", "",
             "| # | Test | Critère | Résultat | Verdict |", "|---|---|---|---|---|"]
    cv_rows(lines)
    certificate_rows(lines)
    certificate_on_truth(lines, compare)
    cross_rows(lines, compare)
    rerun_rows(lines)
    twin_row(lines, compare)
    codes = status["steps"].get("codes_verdict")
    lines.append("| 6 | Verdict des codes (AVL × XFOIL à des sections éloignées) | aucune section éloignée ne "
                 "bat la meilleure | "
                 + ("voir `results/codes_verdict.json`" if codes and codes["state"] == "done"
                    else "non lancé cette nuit") + " | "
                 + ("à lire" if codes and codes["state"] == "done" else "pas de résultat") + " |")
    profile = read(R11 / "profile.json")
    if profile is not None:
        modes = " ; ".join(f"{name} ({fr(100 * m['weight'], 0)} %) : {fr(m['gain_kt'], 2, True)} nd"
                           for name, m in profile["modes"].items() if m["gain_kt"] is not None)
        worst = min(min(m["gain_when_weight_scaled"]) for m in profile["modes"].values())
        lines.append("| 7 | Profil d'usage (run du 11/11) | gain positif pour tout poids de mode entre 0 et "
                     f"3 fois l'actuel | gain {fr(profile['gain_kt'], 3, True)} nd ; par mode : {modes} ; "
                     f"gain minimal sur le balayage {fr(worst, 3, True)} nd | {verdict(worst > 0)} |")
    lines += ["",
              "Lecture des critères : « écart réduit » = erreur de prédiction divisée par l'écart-type "
              "annoncé (1 = incertitude juste, au-dessus = modèle trop sûr de lui). κ_loin est cet écart "
              "réduit à 1 score d'éloignement ; il sert à corriger l'incertitude du certificat. Les lignes "
              "« premier ordre » utilisent les états VPP figés de la meilleure section et sont indicatives "
              "pour des sections éloignées ; les lignes « VPP » sont exactes sur le modèle.", ""] \
        + PURPOSE + ["## Détail", "",
              "- Jumeau numérique : [twin_experiments/README.md](../twin_experiments/README.md) et ses "
              "figures.",
              "- Figures du run du 11/11 : [figures/index.html](figures/index.html) ; du run sans CFD : "
              "[figures/nocfd/index.html](figures/nocfd/index.html).",
              "- Résultats bruts : `results/*.json` (11/11), `results/nocfd/*.json`, "
              "`../twin_experiments/results/*.json`.",
              "- Journaux : `logs/night/` (un fichier par étape, `night.log`, `status.json`, `git.log`).",
              "", "## Déroulement", ""] + steps_table(status) + ["", "## Notes", ""] \
        + [f"- {note}" for note in status["notes"]] + [
            "", "## Limites à garder en tête", "",
            "- Le jumeau a pour vérité un krigeage : il juge l'algorithme, pas la régularité des vrais "
            "codes.",
            "- La validation croisée avec zone d'exclusion retire des données : à grand rayon il en reste "
            "peu, et l'erreur mesurée est celle d'un modèle plus pauvre que le vrai (test sévère).",
            "- Le certificat et le jugement croisé « premier ordre » gèlent les états VPP.",
            "- Aucun calcul CFD ; les codes aéro n'ont pas été rappelés sauf si la ligne 6 le dit.", ""]
    (HERE / "VALIDATION_2026-11-13.md").write_text("\n".join(lines), encoding="utf-8")
    print("VALIDATION_2026-11-13.md written", flush=True)


if __name__ == "__main__":
    main()
