#!/bin/bash
# CFD mesh study of the C-foil, weekend of 2026-10-23 (OpenFOAM v2512, WSL Ubuntu-24.04).
#
# Template: the fine 3D case of 2026-10-20 (cfd3d/83cb4c37e3a33c13_0e2af9, bdfoil_v1 pipeline): same
# STL (C-foil at one attitude, U = 11.832 m/s), kOmegaSST, Spalding wall function, symmetry plane
# "hull" at z = 0, refinement levels (6 7) on the foil, distance and wake-box refinements, 16 MPI
# processes. Changes, so that the meshes form a proper refinement study:
#   * prism layers in ABSOLUTE size (thicknessModel firstAndExpansion, relativeSizes false): the
#     y+ no longer changes with the mesh (it went 217 -> 96 between the two meshes of 2026-10-20);
#   * the refinement ratio comes from the background mesh (blockMesh cells x 1.3): the snappy
#     levels are relative to it, every cell size scales by the same ratio;
#   * 200 first-order iterations then 500 second-order ones (700 in all): the forces of the
#     2026-10-20 runs were converged to 1e-4 at iteration ~380 (they ran 2440-2470).
# Cases, in priority order (a case starts only if its estimated duration fits in the budget):
#   G2, G1, G3: the GCI series at y+ ~ 50 (cell-centre y+ of OpenFOAM), backgrounds x1, /1.3, x1.3;
#   Y100, Y30: the wall resolution at the G2 mesh (y+ ~ 100 and ~ 30);
#   D2: the G1 mesh in a domain twice as large (far-field check).
# First layer thickness for a cell-centre y+: dy = 2 y+ nu / u_tau, u_tau = U sqrt(Cf / 2),
# Cf = 0.0576 Re_x^-0.2 at mid-chord (Re_x = 3.5e6): u_tau = 0.444 m/s, nu = 1.189e-6 m2/s.
#
# Usage: cfd_mesh_study.sh <budget hours> | --validate (builds the cases and checks the set-up
# with blockMesh and snappyHexMesh -dry-run, no meshing, no solving).
MODE=${1:-36}
START=$(date +%s)
# the OpenFOAM bashrc reads unset variables: set -u only after it (as the bdfoil_v1 run.sh)
source /usr/lib/openfoam/openfoam2512/etc/bashrc > /dev/null 2>&1
set -u
TEMPLATE=/home/bdcfd/bdcfd/cfd3d/83cb4c37e3a33c13_0e2af9
WORK=/home/bdcfd/bdcfd/meshstudy_20261023
OUT=/mnt/c/Users/SIM/Desktop/Code-Adri/Multifidelity-GP-Clean/vpp_acceleration/runs/2026-10-23_weekend/cfd
NP=16
mkdir -p "$WORK" "$OUT"
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/cfd.log"; }

#     name            nx ny nz  first[m]  layers scale  est.[h]
CASES="G2_medium_yp50   54 39 25  2.7e-4    8      1      2.0
G1_coarse_yp50   42 30 19  2.7e-4    8      1      1.0
G3_fine_yp50     70 51 33  2.7e-4    8      1      4.5
Y100_medium      54 39 25  5.4e-4    6      1      2.0
Y30_medium       54 39 25  1.6e-4    10     1      2.2
D2_coarse_yp50   84 60 38  2.7e-4    8      2      1.3"

make_case() {   # name nx ny nz first layers scale
    local d="$WORK/$1"
    rm -rf "$d" && mkdir -p "$d/constant" || return 10
    cp -r "$TEMPLATE/0.orig" "$TEMPLATE/system" "$d/" || return 11
    cp -r "$TEMPLATE/constant/triSurface" "$TEMPLATE/constant/transportProperties" \
          "$TEMPLATE/constant/turbulenceProperties" "$d/constant/" || return 12
    cd "$d" || return 13
    sed -i "s/(54 39 25)/($2 $3 $4)/" system/blockMeshDict
    grep -q "($2 $3 $4)" system/blockMeshDict || return 14
    if [ "$7" != "1" ]; then           # domain scaled about the origin (the foil is at x in [-0.7, 0])
        python3 - "$7" system/blockMeshDict <<'PY' || return 15
import re, sys
scale, path = float(sys.argv[1]), sys.argv[2]
text = open(path).read()
def scaled(m):
    points = re.findall(r"\(([-\d.e+ ]+)\)", m.group(1))
    out = " ".join("(" + " ".join(f"{float(v) * scale:.10g}" for v in p.split()) + ")" for p in points)
    return "vertices        (" + out + ");"
text, n = re.subn(r"vertices\s+\((.*?)\);", scaled, text, count=1, flags=re.S)
assert n == 1, "vertices not found"
open(path, "w").write(text)
PY
    fi
    local s=system/snappyHexMeshDict
    foamDictionary $s -entry addLayersControls/relativeSizes -set false > /dev/null || return 16
    foamDictionary $s -entry addLayersControls/thicknessModel -set firstAndExpansion > /dev/null || return 16
    foamDictionary $s -entry addLayersControls/firstLayerThickness -set "$5" > /dev/null || return 16
    foamDictionary $s -entry addLayersControls/expansionRatio -set 1.2 > /dev/null || return 16
    foamDictionary $s -entry addLayersControls/finalLayerThickness -remove > /dev/null || return 16
    foamDictionary $s -entry addLayersControls/minThickness -set "$5" > /dev/null || return 16
    foamDictionary $s -entry addLayersControls/layers/foil/nSurfaceLayers -set "$6" > /dev/null || return 16
    foamDictionary system/controlDict -entry endTime -set 200 > /dev/null || return 17
    foamDictionary system/controlDict -entry writeInterval -set 200 > /dev/null || return 17
    return 0
}

run_case() {    # in the case folder; same steps as the bdfoil_v1 run.sh, 700 iterations in all
    surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 || return 31
    blockMesh > log.blockMesh 2>&1 || return 32
    decomposePar -force > log.decomposePar 2>&1 || return 33
    T_MESH0=$(date +%s)
    mpirun -np $NP snappyHexMesh -parallel -overwrite < /dev/null > log.snappyHexMesh 2>&1 || return 34
    mpirun -np $NP checkMesh -parallel -constant < /dev/null > log.checkMesh 2>&1
    T_MESH1=$(date +%s)
    for p in processor*; do mkdir -p "$p/0" && cp -r 0.orig/. "$p/0/" || return 40; done
    mpirun -np $NP potentialFoam -parallel -writephi < /dev/null > log.potentialFoam 2>&1 || return 41
    cp system/fvSchemes.upwind system/fvSchemes
    mpirun -np $NP simpleFoam -parallel < /dev/null > log.simpleFoam.1 2>&1 || return 42
    cp system/fvSchemes.second system/fvSchemes
    foamDictionary system/controlDict -entry endTime -set 700 > /dev/null || return 43
    foamDictionary system/controlDict -entry writeInterval -set 700 > /dev/null || return 43
    mpirun -np $NP simpleFoam -parallel < /dev/null > log.simpleFoam.2 2>&1 || return 44
    T_SOLVE1=$(date +%s)
    return 0
}

last_dat() {    # last data line of the latest time folder of a function object file
    local f
    f=$(ls -d postProcessing/$1/*/ 2>/dev/null | sort -V | tail -1)
    [ -n "$f" ] && grep -v "^#" "$f/$2" | tail -1
}

results() {     # one CSV line per case
    local name=$1 rc=$2
    local cells layer yplus force moment drift
    cells=$(grep -m1 "cells:" log.checkMesh 2>/dev/null | awk '{print $2}')
    layer=$(grep -E "^foil [0-9]+ " log.snappyHexMesh 2>/dev/null | tail -1 | awk '{print $5","$6}')
    yplus=$(last_dat yPlus1 yPlus.dat | awk '{print $NF","$(NF-1)}')
    force=$(last_dat forces1 force.dat | awk '{print $2","$3","$4","$5","$8}')
    moment=$(last_dat forces1 moment.dat | awk '{print $2","$3","$4}')
    # relative change of the total Fx over the last 200 iterations (iterative convergence)
    drift=$(f=$(ls -d postProcessing/forces1/*/ 2>/dev/null | sort -V | tail -1);
            [ -n "$f" ] && grep -v "^#" "$f/force.dat" | tail -201 | awk 'NR==1{a=$2} END{if (a!=0) printf "%.2e", (($2-a)/a)}')
    echo "$name,$rc,${cells:-},${layer:-,},${yplus:-,},${force:-,,,,},${moment:-,,},${drift:-},$((T_MESH1-T_MESH0)),$((T_SOLVE1-T_MESH1))" >> "$OUT/results.csv"
}

if [ ! -f "$OUT/results.csv" ]; then
    echo "case,rc,cells,layer_thickness_m,layer_percent,yplus_avg,yplus_max,Fx,Fy,Fz,Fx_pressure,Fx_viscous,Mx,My,Mz,Fx_drift_last200,mesh_s,solve_s" > "$OUT/results.csv"
fi

echo "$CASES" | while read -r name nx ny nz first layers scale est; do
    [ -z "$name" ] && continue
    if [ "$MODE" = "--validate" ]; then
        make_case "$name" "$nx" "$ny" "$nz" "$first" "$layers" "$scale" || { log "VALIDATE $name: make_case failed ($?)"; continue; }
        blockMesh > log.blockMesh 2>&1 && cells=$(grep -m1 "nCells" log.blockMesh | awk '{print $2}')
        surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
        snappyHexMesh -dry-run > log.dryrun 2>&1; rc=$?
        log "VALIDATE $name: blockMesh $(grep -c 'End' log.blockMesh) ok, background cells ${cells:-?}, snappy dry-run rc=$rc, layers: $(foamDictionary system/snappyHexMeshDict -entry addLayersControls/firstLayerThickness -value) m x $(foamDictionary system/snappyHexMeshDict -entry addLayersControls/layers/foil/nSurfaceLayers -value)"
        continue
    fi
    elapsed_h=$(( ($(date +%s) - START) / 3600 ))
    if awk -v e="$elapsed_h" -v est="$est" -v b="$MODE" 'BEGIN{exit !(e + est > b)}'; then
        log "$name skipped: estimated $est h, $elapsed_h h used of the budget $MODE h"
        continue
    fi
    log "$name: start (background $nx x $ny x $nz, first layer $first m x $layers, domain x$scale)"
    T_MESH0=$(date +%s); T_MESH1=$T_MESH0; T_SOLVE1=$T_MESH0
    if make_case "$name" "$nx" "$ny" "$nz" "$first" "$layers" "$scale"; then
        run_case; rc=$?
    else
        rc=$?
    fi
    results "$name" "$rc"
    log "$name: done rc=$rc ($(tail -1 "$OUT/results.csv"))"
done

if [ "$MODE" != "--validate" ]; then
    log "all cases done; report"
    WINPY=/mnt/c/Users/SIM/Desktop/Code-Adri/Multifidelity-GP-Clean/.venv-mds/Scripts/python.exe
    "$WINPY" 'C:\Users\SIM\Desktop\Code-Adri\Multifidelity-GP-Clean\vpp_acceleration\runs\2026-10-23_weekend\cfd_mesh_report.py' >> "$OUT/cfd.log" 2>&1
    log "end"
fi
