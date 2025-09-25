#pragma once
#include <sw/compute/ipe.hpp>

namespace sw::compute {

// Base class for modeling Processing Element (PE) in a distributed data path 
template<typename InputType, unsigned NrInputs, typename AccumulationType, typename ResultType>
class KPU_API ProcessingElement : public IProcessingElement {
public:
    ProcessingElement() = default;

    // Data inputs
    void set_input(unsigned var, InputType value) { input[var] = value; }

    // Data outputs (for propagation)
    InputType input(unsigned var) const { return input[var]; }
	ResultType result() const { return static_cast<ResultType>(accumulator); }

    // Process one cycle
    void cycle() {

    }

    // Reset PE state
    void reset() {
        input = 0;
		accumulator = 0;
		accumulating = false;
        last_valid_cycle = 0;
	}

private:

    // Data registers
    InputType input[NrInputs];
    AccumulationType accumulator;

    // Control state
    bool accumulating;
    Cycle last_valid_cycle;
};

}