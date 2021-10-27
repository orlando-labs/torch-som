require_relative 'base'

module Torch
  module NN
    class SelfOrganizedMap < Torch::NN::Module
      module Descent
        class Monotonic < Base
          def initialize(initial:, iterations:)
            raise ArgumentError, "Iterations number must be a positive integer" unless iterations.is_a?(Integer) && iterations > 0
            @iterations = iterations
            
            raise ArgumentError, "Initial value must be a positive integer" unless initial.is_a?(Numeric) && initial > 0
            @initial = initial
            @step_number = 0
          end

          def call(step_number = nil)
            if step_number
              raise ArgumentError, "Step number must be an integer >= 0" unless step_number.is_a?(Integer) && step_number >= 0
              @step_number = step_number
            end
            res = @initial * (1.0 - @step_number.to_f / @iterations)
            @step_number += 1 if @step_number < @iterations
            
            res
          end
        end
      end
    end
  end
end
