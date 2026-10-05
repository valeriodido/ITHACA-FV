# Define line styles for clear distinction
set style line 1 linecolor rgb "#E69F00" lw 2 pt 7 ps 0.6  # Orange
set style line 2 linecolor rgb "#56B4E9" lw 2 pt 5 ps 0.6  # Sky Blue
set style line 3 linecolor rgb "#009E73" lw 2 pt 9 ps 0.6  # Bluish Green
set style line 4 linecolor rgb "#F0E442" lw 2 pt 11 ps 0.6 # Yellow
set style line 5 linecolor rgb "#D55E00" lw 2 pt 15 ps 0.6 # Vermillion
set style line 6 linecolor rgb "#0072B2" lw 2 pt 13 ps 0.6 # Blue

# Output configuration (saves as PNG; change to 'set terminal qt' for an interactive window)
set terminal pngcairo size 1440,1080 font "Helvetica,20"
set output 'resultsGappyPOD.png'

# Prepare a 2 figure multiplot
set multiplot layout 2,1 title "Gappy POD"

# Titles and Axis Labels
set title "Gappy reconstruction error comparison" font ",18"
set xlabel "Sample Index"
set ylabel "L_2 Relative error" font ",18"

# Grid and Visual Styling
set grid
set key top right spacing 1.2
set tics nomirror

# Define base path
dir = "./ITHACAoutput/l2_error/"

# Plotting using zero-based line index ($0) as the X-axis
plot dir."l2_error_2modes_20samples_mat.txt" using 0:1 with linespoints ls 1 title "2 Modes", \
     dir."l2_error_4modes_20samples_mat.txt" using 0:1 with linespoints ls 2 title "4 Modes", \
     dir."l2_error_6modes_20samples_mat.txt" using 0:1 with linespoints ls 3 title "6 Modes", \
     dir."l2_error_8modes_20samples_mat.txt" using 0:1 with linespoints ls 4 title "8 Modes", \
     dir."l2_error_10modes_20samples_mat.txt" using 0:1 with linespoints ls 5 title "10 Modes"


cond_dir = "./ITHACAoutput/condition_numbers/"
set title "Condition numbers for different modes" font ",18"
set xlabel "Number of modes"
set ylabel "Condition Number" font ",18"
# Show ticks only at integer steps of 1
set xtics 1

plot cond_dir."condition_numbers_mat.txt" using 1:2 with linespoints ls 1 title "Condition Number"