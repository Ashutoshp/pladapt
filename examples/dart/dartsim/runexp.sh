for i in `seq 1 50`;
do
    ./run.sh $* --seed=$i
done

#./run.sh --adapt-mgr pmc --decision-horizon 2 --observation-horizon 5 --altitude-levels 4 --accumulate-observations --map-size 20 --num-targets 10 --num-threats 3 --stay-alive-reward 10 --ecm-threat 0.15 --ecm-target 0.35 --error-tolerance 1.0 --destruction-formation-factor=1.75 --seed=$i

#seq 1 50 | parallel --eta -n0 ./run.sh --seed={#} $* | grep ^csv | cut -d, -f2- > r2phase.csv

#for i in `seq 1 1000`;
#do
#    Debug/dartam2 --threat-sensor-fpr=0 --threat-sensor-fnr=0 --target-sensor-fpr=0 --target-sensor-fnr=0 --lookahead-horizon=8
#    Debug/dartam2 --seed=$i
#    Debug/dartam2 --lookahead-horizon=1 --distrib-approx=1 --non-latency-aware
#    Debug/dartam2 --lookahead-horizon=1 --distrib-approx=1 --non-latency-aware --route-length=50 --change-alt-periods=0

#done
