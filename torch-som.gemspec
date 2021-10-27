# frozen_string_literal: true

require_relative "lib/torch/som/version"

Gem::Specification.new do |spec|
  spec.name          = "torch-som"
  spec.version       = Torch::NN::SelfOrganizedMap::VERSION
  spec.authors       = ["Ivan Razuvaev"]
  spec.email         = ["team@orlando-labs.com"]

  spec.summary       = "Self-Organized Map implementation using torch-rb"
  spec.description   = "Self-Organized Map implementation using torch-rb"
  spec.homepage      = "https://github.com/orlando-labs/torch-som"
  spec.license       = "MIT"
  spec.required_ruby_version = Gem::Requirement.new(">= 2.6.0")

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
#  spec.metadata["changelog_uri"] = "TODO: Put your gem's CHANGELOG.md URL here."

  spec.files = Dir.chdir(File.expand_path(__dir__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{\A(?:test|spec|features)/}) }
  end
  spec.require_paths = ["lib"]

  spec.add_dependency "torch-rb", "~> 0.8.0"
end
