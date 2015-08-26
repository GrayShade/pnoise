#!/bin/bash

#echo -e "=== clang -O3:"
#perf stat -r 10 ./bin_test_c_clang 2>&1 > /dev/null | grep time
#echo -e "=== clang2 -O3:"
#perf stat -r 10 ./bin_test2_c_clang 2>&1 > /dev/null | grep time
echo -e "\n=== gcc -O3:"
perf stat -r 10 ./bin_test_c_gcc 2>&1 > /dev/null | grep time
echo -e "\n=== gcc2 -O3:"
perf stat -r 10 ./bin_test2_c_gcc 2>&1 > /dev/null | grep time
#objdump -d bin_test2_c_gcc | wc -l
gdb -batch -ex 'file bin_test2_c_gcc' -ex 'disassemble noise2d_get' | cut -f2 | wc -l
./bin_test2_c_gcc | md5sum
exit 0
echo -e "\n=== mono C#:"
perf stat -r 10 mono -O=float32 bin_test_cs 2>&1 > /dev/null | grep time
echo -e "\n=== mono F#:"
perf stat -r 10 mono -O=float32 bin_test_fs.exe 2>&1 > /dev/null | grep time
echo -e "\n=== D (dmd):"
perf stat -r 10 ./bin_test_d_dmd 2>&1 > /dev/null | grep time
echo -e "\n=== D (ldc2):"
perf stat -r 10 ./bin_test_d_ldc 2>&1 > /dev/null | grep time
echo -e "\n=== D (gdc):"
perf stat -r 10 ./bin_test_d_gdc 2>&1 > /dev/null | grep time
echo -e "\n=== Go gc:"
perf stat -r 10 ./bin_test_go_gc 2>&1 > /dev/null | grep time
echo -e "\n=== Go gccgo -O3:"
perf stat -r 10 ./bin_test_go_gccgo 2>&1 > /dev/null | grep time
echo -e "\n=== Rust:"
perf stat -r 10 ./bin_test_rs 2>&1 >/dev/null | grep time
echo -e "\n=== Nim (gcc):"
perf stat -r 10 ./bin_test_nim_gcc 2>&1 > /dev/null | grep time
echo -e "\n=== Nim (clang):"
perf stat -r 10 ./bin_test_nim_clang 2>&1 > /dev/null | grep time
echo -e "\n=== Crystal:"
perf stat -r 10 ./bin_test_cr 2>&1 > /dev/null | grep time
echo -e "\n=== Java:"
perf stat -r 10 java -cp . test 2>&1 > /dev/null | grep time
