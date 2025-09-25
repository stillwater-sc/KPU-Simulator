#pragma once

namespace sw::compute {

    class IProcessingElement {
    public:
        virtual ~IProcessingElement() = default;
        virtual void reset() = 0;
        virtual void cycle() = 0;
        virtual void start() = 0;
        virtual bool is_busy() const = 0;
    };

}