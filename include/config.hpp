#pragma once

#include <cstdint>
/**
 * To update on every abi update so two you won't be able to link the new declarations against an older library.
 */
#define abi_rev v_1

using byte = uint8_t;
using dword = uint32_t;
using qword = uint64_t;
