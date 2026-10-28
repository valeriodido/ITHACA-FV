# Force coefficients and L2 errors of one or more ROMs, for one parametric run
# Usage:
#   gnuplot -e "cases='D E G'; run='01'" errorPlots.gp
#   gnuplot -e "cases='D E'; run='01'; labels='N=15_separate N=15_stacked'" errorPlots.gp
# Variables:
#   cases    : ROM identifiers (romID in ITHACAdict), separated by spaces
#   run      : parametric run (suffix), default 01
#   labels   : optional legend entries, one per case, '_' is shown as a space (default: "ROM <case>")
#   fomForces: FOM force coefficients, default ITHACAoutput/OnlineFOM<run>/forceCoeffs.dat
#   dtWrite  : time between two written fields, for the time axis of the errors (default 0.2)
# Inputs, written by TestLSPG:
#   ITHACAoutput/ROM/<case>/<run>/forceCoeffs.dat
#   ITHACAoutput/ROM/<case>/<run>/errors/L2ErrorLSPG{U,p,k,omega}_mat.txt   ROM vs FOM
#   ITHACAoutput/ROM/<case>/<run>/errors/L2errorProj{U,p,k,omega}_mat.txt   projection of the FOM on the basis
# Output: images/lift_and_drag_coefficients_<run>.png, images/l2Errors_<run>.png

reset

if (!exists("cases")) {
    print "Set the ROMs to plot: gnuplot -e \"cases='D E'; run='01'\" errorPlots.gp"
    exit
}
if (!exists("run")) run = "01"
if (!exists("dtWrite")) dtWrite = 0.2
if (!exists("fomForces")) fomForces = "ITHACAoutput/OnlineFOM".run."/forceCoeffs.dat"

nCases = words(cases)
caseName(i) = word(cases, i)
romFolder(i) = "ITHACAoutput/ROM/".caseName(i)."/".run
forces(i) = romFolder(i)."/forceCoeffs.dat"
romError(i, f) = romFolder(i)."/errors/L2ErrorLSPG".f."_mat.txt"
projError(i, f) = romFolder(i)."/errors/L2errorProj".f."_mat.txt"
label(i) = exists("labels") ? system("echo '".word(labels, i)."' | tr '_' ' '") : "ROM ".caseName(i)
fexists(f) = system("test -f '".f."' && echo 1 || echo 0") + 0

haveFOM = fexists(fomForces)
if (!haveFOM) print "FOM force coefficients not found (".fomForces."): plotting the ROMs only"

# --- Global Style Definitions ---
colors = "#CC0000 #0066CC #009966 #FF9900 #7733AA #04B5C8 #996633 #E066A0"
color(i) = word(colors, (i - 1) % words(colors) + 1)
set style line 7 lc rgb "#010101" lw 1.5 dt 1 pt 7

set border 15 back lw 1.2 lc rgb "#333333"
set tics font "Arial,14" textcolor rgb "#333333"
set label font "Arial,16" textcolor rgb "#333333"
system("mkdir -p images")

# =====================================================================
# 1. LIFT AND DRAG COEFFICIENTS (2 subplots)
# forceCoeffs columns: 1 time, 2 Cd, 5 Cl
# =====================================================================
set terminal pngcairo size 800,1000 font "Arial,14"
set output "./images/lift_and_drag_coefficients_".run.".png"

set multiplot layout 2,1 title "Lift and drag coefficients, run ".run font "Arial,14"
set xlabel "Time [s]" font "Arial,12"
set key left top font "Arial,12"

do for [col in "5 2"] {
    c = col + 0
    set ylabel (c == 5 ? "C_l" : "C_d") font "Arial,12"
    if (haveFOM) {
        plot for [i = 1:nCases] forces(i) every ::2 using 1:c with linespoints lw 2 lc rgb color(i) title label(i), \
             fomForces every ::2 using 1:c with linespoints ls 7 title "FOM"
    } else {
        plot for [i = 1:nCases] forces(i) every ::2 using 1:c with linespoints lw 2 lc rgb color(i) title label(i)
    }
}

unset multiplot
unset output

# =====================================================================
# 2. L2 RELATIVE ERRORS (2x2: U, p, k, omega)
# Solid: ROM vs FOM. Dashed, same colour: projection error of the FOM on the basis of that ROM
# =====================================================================
set terminal pngcairo size 1000,1000 font "Arial,14"
set output "./images/l2Errors_".run.".png"

set multiplot layout 2,2 title "Relative L_2 errors, run ".run font "Arial,18"
set xlabel "Time [s]" font "Arial,12" offset 0,-0.5
set ylabel "L_2 relative error" font "Arial,12" offset -0.5,0
set key left top font "Arial,10"

fields = "U p k omega"
do for [f = 1:4] {
    fieldName = word(fields, f)
    set title "L_2 relative error of ".fieldName font "Arial,12" offset 0,0.5
    plot for [i = 1:nCases] romError(i, fieldName) using ($0 * dtWrite):1 with linespoints lw 2 lc rgb color(i) \
             title (f == 1 ? label(i) : ""), \
         for [i = 1:nCases] projError(i, fieldName) using ($0 * dtWrite):1 with lines lw 1.5 dt 2 lc rgb color(i) \
             title (f == 1 ? "projection, ".label(i) : "")
}

unset multiplot
unset output
