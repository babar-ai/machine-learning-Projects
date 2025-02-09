document.getElementById('healthForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const data = {
        age: parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        height_cm: parseFloat(document.getElementById('height_cm').value),
        weight_kg: parseFloat(document.getElementById('weight_kg').value),
        body_fat: parseFloat(document.getElementById('body_fat').value),
        diastolic: parseInt(document.getElementById('diastolic').value),
        systolic: parseInt(document.getElementById('systolic').value),
        grip_force: parseFloat(document.getElementById('grip_force').value),
        sit_and_bend_forward_cm: parseFloat(document.getElementById('sit_and_bend_forward_cm').value),
        sit_ups_counts: parseInt(document.getElementById('sit_ups_counts').value),
        broad_jump_cm: parseInt(document.getElementById('broad_jump_cm').value)
    };

    try {
        const response = await fetch('http://127.0.0.1:8000/predicate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        document.getElementById('result').innerText = result.prediction;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred. Please try again.';
    }
});
