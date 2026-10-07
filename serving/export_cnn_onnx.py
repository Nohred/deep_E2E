import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn import CNN

def main() -> None:
    model = CNN(num_classes=10)

    state_dict = torch.load(
        PROJECT_ROOT / "artifacts" / "best_model.pth",
        map_location=torch.device('cpu'),
    )

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(8, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image

    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=True,
    )

    output_path = Path(__file__).resolve().parent / "model_repository" / "cnn" / "1" / "model.onnx"
    onnx_program.save(output_path)

    onnx_model = onnx.load(output_path)
    # Triton 23.06 ships an ONNX Runtime that supports IR versions up to 9.
    onnx_model.ir_version = 9
    onnx.save(onnx_model, output_path)

    ## Validar
    onnx.checker.check_model(onnx_model)
    with torch.inference_mode():
        torch_output = model(dummy_input)

    session = ort.InferenceSession(
        output_path,
        providers=['CPUExecutionProvider']
    )

    onnx_output = session.run(
        ["output"],
        {"input": dummy_input.numpy()}
    )[0]

    torch.testing.assert_close( # Si la salida de Torch y ONNX no son iguales, lanza un error
        torch_output,
        torch.tensor(onnx_output),
        rtol=1e-03,
        atol=1e-05
    )

    print("ONNX export and validation successful!")

if __name__ == "__main__":
    main()
