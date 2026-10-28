# Plots of one LM diagnostic dump (ITHACAdict, lmDiagnostics)
# Usage: gnuplot -e "dump='ITHACAoutput/ROM/<caseID>/<run>/lmDiagnostics/2.85'" lmDiagnostics.gp
# Output: images/lmDiagnostics_<time>.png and images/lmDiagnostics_<time>_coordinates.png
#
# How to read them (see also summary.log in the dump folder):
#   singular values      : a long tail far below 1 means an ill-conditioned Jacobian
#   actual / predicted   : -> 1 as s -> 0 for a smooth residual and a correct Jacobian
#   kink indicator K     : -> 0 linearly in s if smooth; stays O(1) at small s for a kink
#   switches             : k/omega cells crossing the clipping bound, faces where phi changes sign

reset

if (!exists("dump")) {
    print "Set the dump folder: gnuplot -e \"dump='<ROM folder>/lmDiagnostics/<time>'\" lmDiagnostics.gp"
    exit
}
time = system("basename ".dump)

SV = dump."/singularValues.dat"
GN = dump."/direction_gaussNewton.dat"
SD = dump."/direction_steepestDescent.dat"
CMAX = dump."/coordinates_max.dat"
CSMALL = dump."/coordinates_smallStep.dat"

set style line 1 lc rgb "#CC0000" lw 2 pt 7 ps 0.8
set style line 2 lc rgb "#0066CC" lw 2 pt 5 ps 0.8
set style line 3 lc rgb "#009966" lw 2 pt 9 ps 0.8
set style line 4 lc rgb "#FF9900" lw 2 pt 11 ps 0.8
set style line 5 lc rgb "#7733AA" lw 2 pt 13 ps 0.8
set style line 7 lc rgb "#010101" lw 2 pt 3 ps 0.8
set border 15 back lw 1.2 lc rgb "#333333"
set grid lc rgb "#DDDDDD"
set key font "Arial,10"

# =====================================================================
# 1. OVERVIEW (2x2)
# =====================================================================
set terminal pngcairo size 1300,1000 font "Arial,12"
set output "./images/lmDiagnostics_".time.".png"
set multiplot layout 2,2 title "LM diagnostics at t = ".time font "Arial,14"

# --- Singular values of the Jacobian
set title "Singular values of J (scaled coordinates)"
set logscale y
set format y "10^{%L}"
set xlabel "index"
set ylabel "{/Symbol s}_i / {/Symbol s}_{max}"
plot SV using 1:3 with linespoints ls 7 notitle

# --- Local model along the two directions
set title "Actual / predicted reduction of ||r|| along the direction"
set logscale x
unset logscale y
set format y "%g"
set format x "10^{%L}"
set xlabel "step s"
set ylabel "actual / predicted"
set yrange [-2:2]
plot GN using 1:($9/$10) with linespoints ls 1 title "Gauss-Newton", \
     SD using 1:($9/$10) with linespoints ls 2 title "steepest descent", \
     1 with lines ls 7 dt 2 notitle
set autoscale y

# --- Kink indicator per block, max over the coordinates
set title "Kink indicator, max over the coordinates"
set logscale y
set format y "10^{%L}"
set ylabel "K"
set key bottom left
plot CMAX using 1:2 with linespoints ls 7 title "total", \
     CMAX using 1:3 with linespoints ls 1 title "U", \
     CMAX using 1:4 with linespoints ls 2 title "p", \
     CMAX using 1:5 with linespoints ls 3 title "k", \
     CMAX using 1:6 with linespoints ls 4 title "omega", \
     CMAX using 1:7 with linespoints ls 5 title "phi"

# --- Discrete switches within the step
set title "Switches within the step, max over the coordinates"
unset logscale y
set format y "%g"
set ylabel "count"
set key top left
plot CMAX using 1:8 with linespoints ls 3 title "k clipping", \
     CMAX using 1:9 with linespoints ls 4 title "omega clipping", \
     CMAX using 1:10 with linespoints ls 1 title "phi sign"

unset multiplot
unset output

# =====================================================================
# 2. KINK INDICATOR PER COORDINATE AT THE SMALLEST STEP
# =====================================================================
reset
set terminal pngcairo size 1300,600 font "Arial,12"
set output "./images/lmDiagnostics_".time."_coordinates.png"
set title "Kink indicator per reduced coordinate at the smallest step, t = ".time font "Arial,14"
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.8 border -1
set logscale y
set format y "10^{%L}"
set ylabel "K"
set xtics rotate by -60 font "Arial,9"
set grid ytics lc rgb "#DDDDDD"
set key top right font "Arial,10"
plot CSMALL using 3:xtic(2) lc rgb "#010101" title "total", \
     "" using 4 lc rgb "#CC0000" title "U", \
     "" using 5 lc rgb "#0066CC" title "p", \
     "" using 6 lc rgb "#009966" title "k", \
     "" using 7 lc rgb "#FF9900" title "omega"
unset output
