# System

This directory defines the interfaces of the top-level system simulator shell.

A typical system simulator is constructed by creating the constituent components and wiring them up according to the system under test.

For example, if we want to model a single Chip KPU solution, with its own local HBM memory, and connected to a host processor via PCIe,
we would create a system simulator with the following components

1- CPU to model the host processor
2- External Memory connected to the host
3- KPU to model the accelerator
4- Local Memory connected to the KPU
5- PCIe bridge to connect CPU and KPU subsystems to each other

Here we define the system simulator API.