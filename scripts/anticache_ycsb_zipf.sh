cache_dram_ratio=0.94

for dram_budget in 5 0.3
do
run_time=540
log_file="anticache_ycsb_zipf_dram_${dram_budget}gib.log"
# empty log file
cat /dev/null > $log_file
for read_ratio in 100 0
do
for zipf_factor in 0.7 0.8 0.9
do
build/frontend/ycsb_zipf --trunc=1 --ycsb_tuple_count=10000000 --dram_gib=$dram_budget --worker_threads=1 --index_type=AntiCache  --cached_btree_ram_ratio=$cache_dram_ratio --ycsb_read_ratio=$read_ratio --ssd_path=/mnt/disks/nvme/leanstore --run_for_seconds=$run_time --ycsb_request_dist=zipfian --zipf_factor=$zipf_factor --xmerge >> $log_file 2>&1
done
done
done