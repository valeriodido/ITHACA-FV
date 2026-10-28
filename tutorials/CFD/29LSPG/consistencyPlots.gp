# V1 consistency test plots (Carlberg et al. 2011, Prop. 1)
# Usage: gnuplot -e "caseID='E'; run='01'; v1='FOMconverged'" consistencyPlots.gp
#   caseID: ROM identifier (romID in ITHACAdict)
#   run   : parametric run compared with the FOM trajectory (consistencyTest/suffix), default 01
#   v1    : label of the V1 FOM trajectory (FOM, FOMconverged, ...)
# Inputs, written by TestLSPG with consistencyTest { enabled true; }:
#   ITHACAoutput/ROM/<caseID>/V1_<v1>/residual_{exact,projected}.dat : time total U p k omega phi
#   ITHACAoutput/ROM/<caseID>/<run>/residualHistory.dat : time initialTotal finalTotal U p k omega phi ...
# Output: images/consistency_{total,blocks,lm}_<caseID>_<v1>.png
# The phi block is 0 when fluxResidual is off.

reset

if (!exists("caseID")) {
    print "Set the ROM: gnuplot -e \"caseID='E'; run='01'; v1='FOMconverged'\" consistencyPlots.gp"
    exit
}
if (!exists("run")) run = "01"
if (!exists("v1")) v1 = "FOM"

EXACT = "ITHACAoutput/ROM/".caseID."/V1_".v1."/residual_exact.dat"
PROJ = "ITHACAoutput/ROM/".caseID."/V1_".v1."/residual_projected.dat"
ROM = "ITHACAoutput/ROM/".caseID."/".run."/residualHistory.dat"
tag = caseID."_".v1
system("mkdir -p images")

# --- Global Style Definitions ---
set style line 1 lc rgb "#CC0000" lw 2 dt 1
set style line 2 lc rgb "#0066CC" lw 2 dt 1
set style line 3 lc rgb "#009966" lw 2 dt 1
set style line 4 lc rgb "#FF9900" lw 2 dt 4
set style line 7 lc rgb "#010101" lw 1.5 dt 1

set border 15 back lw 1.2 lc rgb "#333333"
set tics font "Arial,14" textcolor rgb "#333333"
set logscale y
set format y "10^{%L}"
set grid ytics mytics lc rgb "#DDDDDD"
set xlabel "Time [s]" font "Arial,12"
set key right top font "Arial,11"

# Restrict the plots to the window of the FOM trajectory (the ROM may run longer)
stats EXACT using 1 nooutput
set xrange [STATS_min:STATS_max]

# Columns of the FOM tables (time total U p k omega phi) and of the ROM history
# (time initialTotal finalTotal U p k omega phi): the ROM column is shifted by one.

# =====================================================================
# 1. TOTAL RESIDUAL
# =====================================================================
set terminal pngcairo size 900,600 font "Arial,14"
set output "./images/consistency_total_".tag.".png"
set title "ROM residual: ROM solution vs FOM states" font "Arial,14"
set ylabel "||R|| (weighted)" font "Arial,12"

plot ROM using 1:3 with lines ls 7 title "ROM (converged)", \
     EXACT using 1:2 with lines ls 1 title "FOM exact", \
     PROJ using 1:2 with lines ls 3 title "FOM projected", \
     ROM using 1:2 with lines ls 4 title "ROM (initial guess)"

unset output

# =====================================================================
# 2. RESIDUAL PER BLOCK (3x2)
# =====================================================================
set terminal pngcairo size 1200,1300 font "Arial,14"
set output "./images/consistency_blocks_".tag.".png"
set multiplot layout 3,2 title "ROM residual per block" font "Arial,14"
set key font "Arial,9"

names = "U p k omega"
do for [b = 1:4] {
    set title word(names, b) font "Arial,12"
    set ylabel "||R_{".word(names, b)."}||" font "Arial,12"
    plot ROM using 1:(column(b + 3)) with lines ls 7 title "ROM (converged)", \
         EXACT using 1:(column(b + 2)) with lines ls 1 title "FOM exact", \
         PROJ using 1:(column(b + 2)) with lines ls 3 title "FOM projected"
}

unset multiplot
unset output

# =====================================================================
# 3. LM CONVERGENCE OF THE ROM (iterations and function evaluations per step)
# =====================================================================
unset logscale y
set format y "%g"
set terminal pngcairo size 900,600 font "Arial,14"
set output "./images/consistency_lm_".tag.".png"
set title "Levenberg-Marquardt effort per time step" font "Arial,14"
set ylabel "Iterations" font "Arial,12"
set y2label "Function evaluations" font "Arial,12"
set y2tics
set key right top font "Arial,11"

plot ROM using 1:9 with linespoints ls 1 axes x1y1 title "LM iterations", \
     ROM using 1:10 with lines ls 2 axes x1y2 title "Function evaluations"

unset output
