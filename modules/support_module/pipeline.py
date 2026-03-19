from pipelines.demo_pipeline import PIPELINE

def register_pipeline(kernel):
    kernel.pipeline_engine.register_pipeline("demo", PIPELINE)
