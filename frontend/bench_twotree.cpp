#define RECORD_SIZE 120

#include <errno.h>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <immintrin.h>
#include <libaio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <span>
#include <thread>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>

#include "Units.hpp"
#include "interface/StorageInterface.hpp"
#include "leanstore/BTreeAdapter.hpp"
#include "leanstore/storage/hashing/LinearHashing.hpp"
#include "leanstore/storage/hashing/LinearHashingWithOverflowHeap.hpp"
#include "leanstore/Config.hpp"
#include "leanstore/LeanStore.hpp"
#include "leanstore/profiling/counters/WorkerCounters.hpp"
#include "leanstore/utils/FVector.hpp"
#include "leanstore/utils/Files.hpp"
#include "leanstore/utils/RandomGenerator.hpp"
#include "leanstore/utils/ScrambledZipfGenerator.hpp"
#include "leanstore/utils/HotspotGenerator.hpp"
#include "leanstore/utils/HotspotZipfGenerator.hpp"
#include "leanstore/utils/SelfSimilarGenerator.hpp"
#include "lsmt/rocksdb_adapter.hpp"
#include "lsmt/upmigration_rocksdb_adapter.hpp"
#include "twotree/PartitionedBTree.hpp"
#include "twotree/TrieBTree.hpp"
#include "twotree/TwoBTree.hpp"
#include "twotree/TwoLSMT.hpp"
#include "twotree/TrieLSMT.hpp"
#include "twotree/ConcurrentTwoBTree.hpp"
#include "twotree/ConcurrentPartitionedBTree.hpp"
#include "twohash/TwoHash.hpp"
#include "anti-caching/AntiCache.hpp"
#include "anti-caching/AntiCacheBTree.hpp"
#include "hash/HashAdapter.hpp"
#include "heap/HeapFileAdapter.hpp"
#include "heap/IndexedHeapAdapter.hpp"
#include "twoheap/TwoIHeap.hpp"

#include "/users/nuoxu333/vmcache/util.h"

using namespace std;
using namespace leanstore;
using YCSBKey = u64;
using YCSBPayload = BytesPayload<RECORD_SIZE>;

void message() {
  // cout << "record_size: " << ((RECORD_SIZE) + sizeof(YCSBKey)) << std::endl;
  // cout << "Date and Time of the run: " << getCurrentDateTime() << std::endl;
  // cout << "lazy migration sampling rate " << FLAGS_cache_lazy_migration << "%" << std::endl;
  // cout << "hot_pages_limit " << hot_pages_limit << std::endl;
  // cout << "effective_page_to_frame_ratio " << effective_page_to_frame_ratio << std::endl;
  // cout << "index type " << FLAGS_index_type << std::endl; 
  // cout << "request distribution " << FLAGS_ycsb_request_dist << std::endl;
  // cout << "dram_gib " << FLAGS_dram_gib << std::endl;
  // cout << "top_component_size_gib " << top_tree_size_gib << std::endl;
  // cout << "wal=" << FLAGS_wal << std::endl;
  // cout << "zipf_factor=" << FLAGS_zipf_factor << std::endl;
  // cout << "ycsb_read_ratio=" << FLAGS_ycsb_read_ratio << std::endl;
  // cout << "ycsb_update_ratio=" << FLAGS_ycsb_update_ratio << std::endl;
  // cout << "ycsb_scan_ratio=" << FLAGS_ycsb_scan_ratio << std::endl;
  // cout << "run_for_seconds=" << FLAGS_run_for_seconds << std::endl;
}

struct DiskStats {
  std::string device;
  double tps;
  double avgReadsPerSecond;
  double avgWritesPerSecond;
  double dscdPerSecond;
  uint64_t totalKbRead;
  uint64_t totalKbWritten;
  uint64_t totalKbDscd;
};

DiskStats parseIOStatOutput(const std::string& output) {
  std::istringstream iss(output);
  string line;

  DiskStats stats;

  while (std::getline(iss, line)) {
      if(line.find("nvme4n1") != string::npos){
        istringstream lineStream(line);
        lineStream >> stats.device >> stats.tps >> stats.avgReadsPerSecond >> stats.avgWritesPerSecond >> stats.dscdPerSecond >> stats.totalKbRead >> stats.totalKbWritten >> stats.totalKbDscd;
        break;
      }
  }

  return stats;
}

// benchmark tuning parameters
DEFINE_int32(ops, 1000, "number of operations");
DEFINE_int32(key_size, 5, "key size");
DEFINE_int32(value_size, 10, "value size");
DEFINE_string(workload, "random_insert", "workload type");
DEFINE_int32(threads, 1, "number of threads");
DEFINE_bool(use_iterative_flush, false, "use iterative flush or recursive flush");
DEFINE_bool(no_read, false, "insertion only workload");
DEFINE_bool(show_snapshot, false, "show tree detail");
DEFINE_int32(prefill, 5000, "fill up before benchmarking");
DEFINE_int32(insertion_threads, 1, "number of insertion thread");

DEFINE_bool(verify, false, "");
DEFINE_string(index_type, "BTree", "");
DEFINE_uint32(cached_btree, 0, "");
DEFINE_uint32(cached_btree_node_size_type, 0, "");
DEFINE_bool(inclusive_cache, false, "");
DEFINE_uint32(update_or_put, 0, "");
DEFINE_uint32(cache_lazy_migration, 100, "lazy upward migration sampling rate(%)");

std::atomic<bool> keepRunning = {false};
std::atomic<uint64_t> curOps = {0};
std::atomic<uint64_t> totalCnt = {0};

std::string gen_random(const int len) {
  static const char alphanum[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  std::string tmp_s;
  tmp_s.reserve(len);

  for (int i = 0; i < len; ++i) {
    tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
  }

  return tmp_s;
}

inline void showProgress(int i, bool detailed = false) {
  int cond = 0;
  if (i < 1000) {
    cond = 100;
  } else if (i < 10000) {
    cond = 1000;
  } else if (i < 100000) {
    cond = 10000;
  } else if (i < 1000000) {
    cond = 100000;
  } else {
    cond = 100000;
  }
  if(detailed) std::cout << "\rops finished: " << i;
  else if (i % cond == 0 && i >= cond) std::cout << "\rops finished: " << i;
  fflush(stdout);
}

void randomInsertion(std::vector<std::pair<std::string, std::string>>& test) {
  std::cout << "run random insertion" << std::endl;
  // generate testing data
  for(int i = 0; i < FLAGS_ops; ++i) {
    // build key
    // std::string key_str = gen_random(FLAGS_key_size);
    std::string value_str = gen_random(FLAGS_value_size);
    test.push_back(std::make_pair("", value_str));
  }
  std::cout << "done generating data" << std::endl;

  if(FLAGS_insertion_threads < FLAGS_threads) FLAGS_insertion_threads = FLAGS_threads;

  auto start = std::chrono::high_resolution_clock::now();

  keepRunning.store(true);
  parallel_for(0, FLAGS_ops - FLAGS_prefill, FLAGS_insertion_threads,
              [&](uint64_t worker, uint64_t begin, uint64_t end) {
                for (u64 i = begin + 1; i <= end; i++) {
                  curOps.fetch_add(1);
                  totalCnt.fetch_add(1);
                }
              });
  std::cout << "done insertion" << std::endl;
  keepRunning.store(false);
  auto end = std::chrono::high_resolution_clock::now();
  long long duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  float throughput = ((float)FLAGS_ops / duration);
  std::cout << "write throughput: " << throughput << " ops/μs" << std::endl;
} 

void workloadInsertion(std::vector<std::pair<std::string, std::string>>& test) {
  test.reserve(FLAGS_ops);
  if (FLAGS_workload == "random_insert") {
    randomInsertion(test);
  } else if (FLAGS_workload == "sequential_insert") {
    // sequentialInsertion(test);
  } else {
    std::cout << "workload not supported" << std::endl;
    exit(1);
  }
}

void WorkloadLookup() {
  auto start = std::chrono::high_resolution_clock::now();
  {
    keepRunning.store(true);
    parallel_for(0, FLAGS_prefill, FLAGS_threads,
              [&](uint64_t worker, uint64_t begin, uint64_t end) {
                for (u64 i = begin + 1; i <= end; i++) {
                  curOps.fetch_add(1);
                  totalCnt.fetch_add(1);
                }
              });
    keepRunning.store(false);
  }
  auto end = std::chrono::high_resolution_clock::now();
  long long duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  float throughput = ((float)FLAGS_ops / duration);
  std::cout << "read throughput: " << throughput << " ops/μs" << std::endl;
}

void WorkloadInsertion() {
  std::vector<std::pair<std::string, std::string>> keys;
  workloadInsertion(keys);
}

void WorkloadUpdateWrapper() {
  auto start = std::chrono::high_resolution_clock::now();
  keepRunning.store(true);
  parallel_for(0, FLAGS_ops, FLAGS_threads,
            [&](uint64_t worker, uint64_t begin, uint64_t end) {
              for(int i = begin + 1; i <= end; ++i) {
                totalCnt.fetch_add(1);
                curOps.fetch_add(1);
              }                
            });
  keepRunning.store(false);
  auto end = std::chrono::high_resolution_clock::now();
  long long duration =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();
  float throughput = ((float)FLAGS_ops / duration);
  std::cout << "update throughput: " << throughput << " ops/μs" << std::endl;
}

void WorkloadReadModifyWrite() {
  // generate new value for update, keep same length for now.
  // for(int i = 0; i < keys.size(); ++i) {
  //   keys[i].second = gen_random(FLAGS_value_size);
  // }
  auto start = std::chrono::high_resolution_clock::now();
  keepRunning.store(true);
  parallel_for(0, FLAGS_threads, FLAGS_threads,
            [&](uint64_t worker, uint64_t begin, uint64_t end) {
              while(totalCnt.load() < FLAGS_ops) {
                totalCnt.fetch_add(1);
                curOps.fetch_add(1);
              }
            });
  keepRunning.store(false);
  auto end = std::chrono::high_resolution_clock::now();
  long long duration =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();
  float throughput = ((float)FLAGS_ops / duration);
  std::cout << "rmw throughput: " << throughput << " ops/μs" << std::endl;
}

void Run() {
  if(FLAGS_workload == "update") {
    WorkloadUpdateWrapper();
  } else if(FLAGS_workload == "rmw") {
    WorkloadReadModifyWrite();
  } else if(FLAGS_workload == "random_lookup") {
    std::cout << "> " << FLAGS_prefill << " lookups on " << FLAGS_prefill << " key-value pairs" << std::endl;
    WorkloadLookup();
  } else {
    std::cout << "> " << FLAGS_ops - FLAGS_prefill << " insertions after " << FLAGS_prefill << " key-value pairs bulkloaded" << std::endl;
    WorkloadInsertion();
  }
}

DiskStats getDiskStat() {
  string command = "iostat -d -k 1 1 | grep nvme4n1"; // Run iostat command
  string output;
  
  // Execute command and capture output
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) {
      cerr << "Error executing iostat command" << endl;
      return DiskStats{};
  }
  char buffer[128];
  while (!feof(pipe)) {
      if (fgets(buffer, 128, pipe) != NULL) {
          output += buffer;
      }
  }
  pclose(pipe);

  // Parse output
  DiskStats stats = parseIOStatOutput(output);
  return stats;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto statFn = [&]() {
    u64 cnt = 0;
    while(!keepRunning.load());
    cout << "ts,rmb,wmb,ops" << endl;
    while(keepRunning.load()) {
      sleep(1);
      float rmb = 0;
      float wmb = 0;
      int ops = curOps;
      cout << cnt++ << "," << rmb << "," << wmb << "," << ops << endl;
    }
  };

  double effective_page_to_frame_ratio = sizeof(leanstore::storage::BufferFrame::Page) / (sizeof(leanstore::storage::BufferFrame) + 0.0);
  s64 hot_pages_limit = FLAGS_dram_gib * FLAGS_top_component_dram_ratio * 1024 * 1024 * 1024 / sizeof(leanstore::storage::BufferFrame);
  double top_tree_size_gib = FLAGS_dram_gib * FLAGS_top_component_dram_ratio * effective_page_to_frame_ratio;

  LeanStore db;
  db.getBufferManager().hot_pages_limit = hot_pages_limit;
  leanstore::storage::btree::BTreeLL* btree_ptr = nullptr;
  leanstore::storage::btree::BTreeLL* btree2_ptr = nullptr;
  leanstore::storage::hashing::LinearHashTable* ht_ptr = nullptr;
  leanstore::storage::hashing::LinearHashTable* ht2_ptr = nullptr;
  leanstore::storage::hashing::LinearHashTableWithOverflowHeap* htoh2_ptr = nullptr;
  leanstore::storage::heap::HeapFile* hf_ptr = nullptr;
  leanstore::storage::heap::HeapFile* hf2_ptr = nullptr;
  db.getCRManager().scheduleJobSync(0, [&](){
    btree_ptr = &db.registerBTreeLL("btree", true);
    btree2_ptr = &db.registerBTreeLL("btree_cold");
    ht_ptr = &db.registerHashTable("ht", true);
    ht2_ptr = &db.registerHashTable("ht_cold");
    hf_ptr = &db.registerHeapFile("hf", true);
    hf2_ptr = &db.registerHeapFile("hf_cold");
    auto heap_file_for_hashing = &db.registerHeapFile("hf_for_ht_cold");
    htoh2_ptr = &db.registerHashTableWOH("htoh_cold", *heap_file_for_hashing);
  });
  unique_ptr<StorageInterface<YCSBKey, YCSBPayload>> adapter;
  adapter.reset(new ConcurrentTwoBTreeAdapter<YCSBKey, YCSBPayload>(*btree_ptr, *btree2_ptr, top_tree_size_gib, FLAGS_inclusive_cache, FLAGS_cache_lazy_migration));
  // db.startProfilingThread();
  adapter->set_buffer_manager(db.buffer_manager.get());

  thread statThread(statFn);

  // DiskStats startStat = getDiskStat();
  // Run();
  // DiskStats endStat = getDiskStat();

  // std::cout << "Total MB read from disk: " << (endStat.totalKbRead - startStat.totalKbRead) / 1024 << std::endl;
  // std::cout << "Total MB written to disk: " << (endStat.totalKbWritten - startStat.totalKbWritten) / 1024 << std::endl;

  statThread.join();
}