### Submit the sbatch array command to do omnical
#SBATCH --account=jpober-condo
#obs_file_name='./obspoint0'
obs_file_name='obspoint2'
poscal='PhaseII_cal'
mem='80G'
time='10:00:00'

##Read the obs file and put into an array, skipping blank lines if they exist
i=0
while read line
do
   if [ ! -z "$line" ]; then
      obs_id_array[$i]=$line
      i=$((i + 1))
   fi
done < "$obs_file_name"

##Create a list of observations using the specified range, or the full observation id file. 
unset good_obs_list
for obs_id in "${obs_id_array[@]}"; do
     good_obs_list+=($obs_id)
done


N=${#good_obs_list[@]}                    
#sbatch -p batch --array=0-$(($N - 1)) --mem=$mem -t $time -n 8 --export=N=$N,poscal=$poscal, noisecheck.sh ${good_obs_list[@]}
sbatch --account=jpober-condo --array=0-$(($N - 1))%12 --mem=$mem -t $time -n 8 --export=N=$N,poscal=$poscal, ssins.sh ${good_obs_list[@]}
