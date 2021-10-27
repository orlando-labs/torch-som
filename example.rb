require 'optparse'
require 'torch/som'
require 'csv'
require 'matplotlib/pyplot'
require 'numpy'
require 'tqdm'

options = {
  x: 20,
  y: 10,
  iterations: 100,
  output: 'output.png'
}

optparser = OptionParser.new do |opts|
  opts.banner = "bundle exec ruby #{__FILE__} [OPTIONS]"
  opts.on('-x X', Integer, "X resolution of map (default: #{options[:x]})")
  opts.on('-y Y', Integer, "Y resolution of map (default: #{options[:x]})")
  opts.on('-o', '--output FILE', "Output map picture filename (default: #{options[:output]})")
  opts.on('-n', '--iterations NUM', Integer, "Number of iterations (default: #{options[:iterations]})")
end.parse!(into: options)

input = [
  [0.0, 0.0, 0.0],
  [0.0, 0.0, 1.0],
  [0.0, 0.0, 0.5],
  [0.125, 0.529, 1.0],
  [0.33, 0.4, 0.67],
  [0.6, 0.5, 1.0],
  [0.0, 1.0, 0.0],
  [1.0, 0.0, 0.0],
  [0.0, 1.0, 1.0],
  [1.0, 0.0, 1.0],
  [1.0, 1.0, 0.0],
  [1.0, 1.0, 1.0],
  [0.33, 0.33, 0.33],
  [0.5, 0.5, 0.5],
  [0.66, 0.66, 0.66]
]

som = Torch::NN::SOM.new(
  options[:x], 
  options[:y], 
  dim: input.first.size, iterations: options[:iterations]
)
som.fit input
#som.fit input, progress: [:with_progress, {desc: "Fitting"}]

plt = Matplotlib::Pyplot
plt.imshow Numpy.asarray(som.weights_2d.to_a)
plt.savefig options[:output]
