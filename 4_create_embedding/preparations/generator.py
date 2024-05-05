import generator.skip_grams_gen as gen


def execute(gen_specs: gen.SkipGenSpecs) -> gen.SkipGramsGenerator:
    generator = gen.SkipGramsGenerator(gen_specs)
    return generator
