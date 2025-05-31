UNDERSAMPLINGWORKDIR="/home/User/ACL2025_Undersampling/"

cd "$UNDERSAMPLINGWORKDIR"

datain="$UNDERSAMPLINGWORKDIR/resources/datasets"
out="$UNDERSAMPLINGWORKDIR/resources/outunder"

mkdir -p $out

datasets=(sentistrength_myspace_2L) 
methods=(tl sbc renn oss obu nearmiss_1 nearmiss_2 nearmiss_3 ncr iht enn cnn cc_nn allknn enub enut enuc enur akcs e2sc_us ubr)
# methods=(e2sc_us ubr)

pwd

for d in ${datasets[@]};
do
    echo $d ; 
    for method in ${methods[@]} 
    do
        echo $method ;
        python3.6 run\_undersampling.py -d $d -m $method --datain $datain --out $out;
    done;
done;