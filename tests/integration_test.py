import subprocess
import os
import glob
import csv
import math

def test_dedisperse():
    # Define paths
    script_path = os.path.abspath('../pytorch_dedispersion/dedisperse_candidates.py')
    config_path = os.path.abspath('test_config.json')
    output_dir = os.path.abspath('.')

    # Run the script with the test config
    result = subprocess.run(
        ['python', script_path, '-c', config_path, '--gpu', '0', '-v'],
        capture_output=True,
        text=True
    )

    # Print output
    print(result.stdout)

    # Verify the result (check if the candidates file is created)
    expected_output_file = os.path.join(output_dir, 'candidates_*.csv')
    files = glob.glob(expected_output_file)

    assert len(files) == 1, "Expected one candidates file to be created."
    print(f"Test passed. Candidates file created: {files[0]}")

    # Verify the highest SNR line
    expected_snr = 13.897
    expected_time = 2.0288829375
    found_line = False

    with open(files[0], mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            snr = float(row[0])
            time_sec = float(row[2])
            if math.isclose(snr, expected_snr, rel_tol=1e-2) and math.isclose(time_sec, expected_time, rel_tol=1e-5) and row[1] == '1602' and row[3:] == ['1', '475.0']:
                found_line = True
                break

    assert found_line, f"Expected line not found. Closest SNR found: {snr}, Closest Time found: {time_sec}"
    print(f"Test passed. Expected line found: SNR={snr}, Time={time_sec}")
    # Remove the created candidates file
    os.remove(files[0])
    print("All tests passed. Candidates file removed.")

if __name__ == "__main__":
    test_dedisperse()
