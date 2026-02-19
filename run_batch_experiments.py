import subprocess

# List of datasets to process
datasets = ['VV', 'DICM', 'NPE', 'MEF', 'LOL', 'LOLv2real', 'LOLv2synthetic', 'LIME', 'NPE-ex1', 'NPE-ex2', 'NPE-ex3']
iterations = [1,2,3,4,5, 6]

# Color spaces to evaluate
color_spaces = ['hsv', 'rgb']

# Path to the main enhancement script
SCRIPT_PATH = 'inference.py'  # replace with your actual script name if needed

for color_space in color_spaces:
    for iteration in iterations:
        for dataset in datasets:
            # Construct the model name (e.g., FHSV2, FRGB3, etc.)
            if color_space == 'hsv':
                model_prefix = 'THSV'
            elif color_space == 'rgb':
                model_prefix = 'TRGB'
            else:
                model_prefix = 'SBLEND'
            model_name = f"{model_prefix}{iteration}"

            print(f"Running: Dataset={dataset}, ColorSpace={color_space}, Iteration={iteration}, Model={model_name}")

            # Run the enhancement script via subprocess
            cmd = [
                'python', SCRIPT_PATH,
                '--method', 'data',
                '--dataset', dataset,
                '--iteration', str(iteration),
                '--device', 'cuda',
                '--color_space', color_space,
                '--model', model_name
            ]

            subprocess.run(cmd)