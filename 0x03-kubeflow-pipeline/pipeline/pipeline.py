"""
contain the training pipeline compiler file
compile using:
dsl-compile --py pipeline.py --output pipeline.tar.gz
"""
import kfp
import kfp.compiler


@kfp.dsl.component
def training(epochs: int,
             batch_size: int):
    ...
    return kfp.dsl.ContainerOp(
        name="Trianing",
        image="gcr.io/otherproject-294618/oceano:latest",
        command=[
            "python",
            "-m",
            "training.main",
            "--epochs",
            f"{epochs}",
            "--batch_size",
            f"{batch_size}",
        ],
        file_outputs={
            "mlpipeline-metrics": "/mlpipeline-metrics.json",
            "mlpipeline-ui-metadata": "/mlpipeline-ui-metadata.json",

        },
        output_artifact_paths={
            "mlpipeline-metrics": "/mlpipeline-metrics.json",
            "mlpipeline-ui-metadata": "/mlpipeline-ui-metadata.json",
        },
    )


@kfp.dsl.pipeline(name="training_pipeline",
                  description="single step pipeline for training")
def pipeline(epochs: int = 200, batch_size: int = 125):
    training_step = training(
        epochs=epochs,
        batch_size=batch_size
    )


kfp.compiler.Compiler().compile(pipeline, "pipeline.yaml")
