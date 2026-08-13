#!/usr/bin/env bash
# Exercise the public line-oriented supervisor with a repeatable startup file.
set -euo pipefail

assembler=$1
runner=$2
root=$3
prefix=$4

hello_wvm=${prefix}.hello.wvm
graphics_wvm=${prefix}.graphics.wvm
startup=${prefix}.startup.wvs
log=${prefix}.log

"${assembler}" "${root}/programs/hello.wva" -o "${hello_wvm}"
"${assembler}" "${root}/programs/graphics.wva" -o "${graphics_wvm}"

printf '%s\n' \
  "program load hello ${hello_wvm}" \
  "program load graphics ${graphics_wvm}" \
  "launch 3" \
  "vm create hello 64" \
  "vm create graphics 64" \
  "vm engine 1 interpreted" \
  "vm start 0" \
  "wait 0 HALTED 2000" \
  "vm start 1" \
  "wait 1 RUNNING 2000" \
  "vm stop 1" \
  "wait 1 STOPPED 2000" \
  "mailbox 1" \
  "vm resume 1" \
  "wait 1 RUNNING 2000" \
  "vm stop 1" \
  "vm reset 1" \
  "vm delete 0" \
  "vm create hello 64" \
  "vm start 2" \
  "wait 2 HALTED 2000" \
  "vm program 2 graphics" \
  "status 2" \
  "list" \
  "vm delete 1" \
  "vm delete 2" \
  "program unload hello" \
  "program unload graphics" \
  "program list" \
  "quit" > "${startup}"

"${runner}" supervise --script "${startup}" | tee "${log}"

grep -q "population launched: capacity=3 programs=2" "${log}"
grep -q "vm created: vm_id=0 slot=0 program=hello state=READY" "${log}"
grep -q "vm created: vm_id=1 slot=1 program=graphics state=READY" "${log}"
grep -q "wait complete: vm_id=1 state=STOPPED" "${log}"
grep -q "vm 1 mailbox: owner=1" "${log}"
grep -q "vm created: vm_id=2 slot=0 program=hello state=READY" "${log}"
grep -q "vm rebound: vm_id=2 program=graphics state=READY" "${log}"
grep -q "vm 2: slot=0 program=graphics engine=INTERPRETED lifecycle=READY" "${log}"
grep -q "program unloaded: id=0 name=hello" "${log}"
grep -q "program unloaded: id=1 name=graphics" "${log}"

echo "supervisor_startup PASS"
