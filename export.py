import argparse

from evolve import export, Morph

parser = argparse.ArgumentParser()
parser.add_argument("--morph", help="protomorph to mutate", default="base.adam")
args = parser.parse_args()

export(Morph(0, args.morph))