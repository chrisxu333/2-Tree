cache_dram_ratio=0.7

for dram_budget in 5 0.3
do
if [ $dram_budget -eq 4 ]
then
   run_time=400
else
   run_time=540
fi
log_file="lsmt_ycsb_put_only_zipf_dram_${dram_budget}gib.log"
# empty log file
cat /dev/null > $log_file
for read_ratio in 0
do
for zipf_factor in 0.7 0.8 0.9
do
build/frontend/ycsb_zipf --trunc=1 --ycsb_tuple_count=10000000 --dram_gib=$dram_budget --worker_threads=1 --index_type=LSMT  --cached_btree_ram_ratio=$cache_dram_ratio --ycsb_read_ratio=$read_ratio --ssd_path=/mnt/disks/nvme/leanstore --run_for_seconds=$run_time --ycsb_request_dist=zipfian --zipf_factor=$zipf_factor --xmerge --update_or_put=1 >> $log_file 2>&1
done
done
done

for dram_budget in 5 0.3
do
if [ $dram_budget -eq 4 ]
then
   run_time=400
else
   run_time=540
fi
log_file="2lsmt_ycsb_put_only_zipf_dram_${dram_budget}gib.log"
# empty log file
cat /dev/null > $log_file
for read_ratio in 0
do
for zipf_factor in 0.7 0.8 0.9
do
build/frontend/ycsb_zipf --trunc=1 --ycsb_tuple_count=10000000 --dram_gib=$dram_budget --worker_threads=1 --index_type=2LSMT-CF  --cached_btree_ram_ratio=$cache_dram_ratio --ycsb_read_ratio=$read_ratio --ssd_path=/mnt/disks/nvme/leanstore --run_for_seconds=$run_time --ycsb_request_dist=zipfian --cache_lazy_migration=50 --inclusive_cache --zipf_factor=$zipf_factor --xmerge --update_or_put=1 >> $log_file 2>&1
done
done
done