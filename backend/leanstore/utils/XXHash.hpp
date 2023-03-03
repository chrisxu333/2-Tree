#pragma once
#include "Units.hpp"
#include "xxhash.hpp"
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
namespace leanstore
{
namespace utils
{
// -------------------------------------------------------------------------------------
class XXH
{
  public:
   //static u64 hash(u64 val);
   static u64 hash(const u8* d, u16 len);
};
// -------------------------------------------------------------------------------------
}  // namespace utils
}  // namespace leanstore
