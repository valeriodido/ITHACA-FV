reset

set style line 1 lc rgb "#CC0000" lw 2 dt 1   # Crimson line
set style line 2 lc rgb "#0066CC" lw 2 dt 1   # Vibrant Orange line
set style line 3 lc rgb "#009966" lw 2 dt 1   # Teal Green line
set style line 4 lc rgb "#CC0000" lw 2 dt 4   # Crimson line
set style line 5 lc rgb "#0066CC" lw 2 dt 4   # Vibrant Orange line
set style line 6 lc rgb "#009966" lw 2 dt 4   # Teal Green line
set style line 7 lc rgb "#010101" lw 1.5 dt 1
set style line 8 lc rgb "#FF9900" lw 1.5 dt 1   # Deep Blue line

# set style line 4 lc rgb "#FF9900" lw 2.5 dt 2   # Deep Blue line
# set style line 6 lc rgb "#FF00FF" lw 2.5 dt 1

set border 15 back lw 1.2 lc rgb "#333333"

set tics font "Arial,14" textcolor rgb "#333333"
set label font "Arial,16" textcolor rgb "#333333"
set key top right font "Arial,14" box lw 0.8 lc rgb "#CCCCCC" opaque

# L_2 error U - no gif
set terminal pngcairo size 800,600 font "Arial,14"
set output "./images/l2Error_U.png"

set xlabel "Time [s]" font "Arial,12" offset 0,-0.5
set ylabel "Relative error" font "Arial,12" offset -0.5,0
set title "Relative error of the velocity field" font "Arial,12" offset 0,0.5

A = "ITHACAoutput/l2error/01A/L2ErrorLSPGU_mat.txt"
B = "ITHACAoutput/l2error/01B/L2ErrorLSPGU_mat.txt"

plot A using 1 with lines ls 1 title "No BC treatment - 8 Modes", \
     B using 1 with lines ls 3 title "No BC treatment - 10 Modes"

# L2 error p - no gif
set terminal pngcairo size 800,600 font "Arial,14"
set output "./images/l2Error_p.png"

set xlabel "Time [s]" font "Arial,12" offset 0,-0.5
set ylabel "Relative error" font "Arial,12" offset -0.5,0
set title "Relative error of the pressure field" font "Arial,12" offset 0,0.5


A = "ITHACAoutput/l2error/01A/L2ErrorLSPGp_mat.txt"
B = "ITHACAoutput/l2error/01B/L2ErrorLSPGp_mat.txt"

plot A using 1 with lines ls 1 title "No BC treatment - 8 Modes", \
     B using 1 with lines ls 3 title "No BC treatment - 10 Modes"

# L2 error nut - no gif
set terminal pngcairo size 800,600 font "Arial,14"
set output "./images/l2Error_nut.png"

set xlabel "Time [s]" font "Arial,12" offset 0,-0.5
set ylabel "Relative error" font "Arial,12" offset -0.5,0
set title "Relative error of the nut field" font "Arial,12" offset 0,0.5


A = "ITHACAoutput/l2error/01A/L2ErrorLSPGnut_mat.txt"
B = "ITHACAoutput/l2error/01B/L2ErrorLSPGnut_mat.txt"

plot A using 1 with lines ls 1 title "No BC treatment - 8 Modes", \
    B using 1 with lines ls 3 title "No BC treatment - 10 Modes"

