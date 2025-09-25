#pragma once
#include <stdexcept>
#include <sw/concepts.hpp>
#include <sw/compute/ipe.hpp>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

namespace sw::compute {

    // DOT Product Accumulator Processing Element (PE) for systolic array
    template<typename InputType, typename AccumulationType, typename ResultType, unsigned Latency = 1>
    class KPU_API DotProductAccumulator : public IProcessingElement {
    public:
        DotProductAccumulator() = default;
        ~DotProductAccumulator() override = default;

        // modifiers

        // Reset PE state
        void reset() override {
            a = 0;
            b = 0;
            c = 0;
            accumulating = false;
            current_cycle = 0;
        }

        void load_input(unsigned var, InputType value) {
            input(var, value);
        }
        void load_accumulator(AccumulationType value) {
            c = value;
        }
        void start() override {
            accumulating = true;
            current_cycle = 0;
            last_valid_cycle = 0;
        }

        // Process one cycle
        void cycle() override {
            ++current_cycle;
            c += a * b;
            if (current_cycle >= Latency) {
                accumulating = false;
                last_valid_cycle = current_cycle;
            }
        }

        // selectors

        // Check if PE is busy processing
        bool is_busy() const override {
            return accumulating;
        }  
        // Read data registers in the PE
        void input(unsigned var, InputType value) {
            switch (var) {
                case 0: // Input A
                    a = value;
                    break;
                case 1: // Input B
                    b = value;
                    break;
                case 2: // Accumulator reset
                    c = 0;
                    break;
                default:
                    throw std::out_of_range("Invalid input variable index");
            }
            accumulating = true;
        }
 
        // Get the result from the PE
        ResultType result() const {
            return static_cast<ResultType>(c);
        }

    private:

        // Data registers
        InputType a, b;
        AccumulationType c;

        // Control state
        bool accumulating;
        sw::kpu::Cycle current_cycle;
        sw::kpu::Cycle last_valid_cycle;
    };

}  // namespace sw::compute


#ifdef _MSC_VER
#pragma warning(pop)
#endif