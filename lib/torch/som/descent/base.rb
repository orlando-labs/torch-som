require 'torch-rb'

module Torch
  module NN
    class SelfOrganizedMap < Torch::NN::Module
      module Descent
        class Base
          attr_reader :iterations

          def call(step_number = nil)
            raise NotImplementedError
          end
        end
      end
    end
  end
end
