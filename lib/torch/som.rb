# frozen_string_literal: true

require_relative "som/version"
require_relative "som/descent/monotonic"

module Torch
  module NN
    class SelfOrganizedMap < Torch::NN::Module
      attr_reader :weights, :node_coordinates

      def initialize(
        x, y, 
        dim:,
        alpha: nil, sigma: nil,
        iterations: nil
      )
        super()
        
        raise ArgumentError, "X must be a positive integer" unless x.is_a?(Integer) && x > 0
        @x = x

        raise ArgumentError, "Y must be a positive integer" unless y.is_a?(Integer) && y > 0
        @y = y

        raise ArgumentError, "Dimension must be a positive integer" unless dim.is_a?(Integer) && dim > 0
        @dim = dim

        unless alpha.nil? or alpha.is_a?(Descent::Base)
          raise ArgumentError, "alpha(t) must be a Descent::Base subclass"
        end
        
        unless sigma.nil? or sigma.is_a?(Descent::Base)
          raise ArgumentError, "sigma(t) must be a Descent::Base subclass"
        end
        
        if iterations.nil? && (alpha.nil? or sigma.nil?)
          raise ArgumentError, "Iterations number must be provided if no alpha(t) or sigma(t) given"
        end

        if iterations && (alpha or sigma)
          raise ArgumentError, "Steps must not be provided if alpha(t) or sigma(t) given"
        end

        if alpha&.iterations && sigma&.iterations && alpha.iterations != sigma.iterations
          raise ArgumentError, "alpha(t) and sigma(t) are designed for different iterations count"
        end

        @steps = alpha&.iterations || sigma&.iterations || iterations

        @alpha_t = alpha || Descent::Monotonic.new(initial: 0.25, iterations: @steps)
        @sigma_t = sigma || Descent::Monotonic.new(initial: [@x, @y].max / 2.0, iterations: @steps)

        @weights = Torch.rand(@x * @y, @dim)
        @meter = Torch::NN::PairwiseDistance.new

        @node_coordinates = Torch.tensor(@x.times.to_a.product(@y.times.to_a), dtype: :long)
      end

      def forward(x, step_number = nil)
        x = tensor!(x)
        input = Torch.stack(Array.new(@x * @y) { x })
        
        bmu_1d_index = @meter.(input, @weights).argmin(dim: 0).item
        bmu_2d_index = Torch.tensor([bmu_1d_index.div(@y), bmu_1d_index % @y])
        
        alpha = @alpha_t.(step_number)
        sigma = @sigma_t.(step_number)

        dists = Torch.stack(Array.new(@x * @y) { bmu_2d_index }) - @node_coordinates
        sq_dists = (dists * dists).sum(dim: 1)
        
        h = (sq_dists / 2.0 / sigma / sigma).neg.exp * alpha

        delta = Torch.einsum 'ij,i->ij', [input - @weights, h]
        @weights += delta
      end

      def locations_for(vectors)
        vectors.map do |x|
          bmu_1d_index = @meter.(tensor!(x), @weights).argmin(dim: 0).item
          [bmu_1d_index.div(@y), bmu_1d_index % @y]
        end
      end

      def fit(vectors, progress: [:itself])
        @steps.times.public_send(*progress).each do |i|
          vectors.each do |x|
            forward x, i
          end
        end
        
        locations_for vectors
      end

      def weights_2d
        @weights.view(@x, @y, @dim)
      end

      private
      def tensor!(x)
        x.is_a?(Tensor) ? x : Torch.tensor(x)
      end
    end
    
    SOM = SelfOrganizedMap
  end
end
