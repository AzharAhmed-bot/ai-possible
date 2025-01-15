from model import model
import numpy as np

# Observed data
#  "umbrella","umbrella","no umbrella","umbrella","umbrella","umbrella","umbrella","no umbrella","no umbrella"
observations = np.array([[[0],[0],[1],[0],[0],[0],[0],[1],[1]]])
predictions = model.predict(observations)

weather_map={0:"Sunny",1:"Rainy"}
umbrella_map={0:"Carry umbrella",1:"Did not carry umbrella"}

print("Observation and predicted weather states: ")
for observation,prediction in zip(observations[0],predictions[0]):
    umbrella_status=umbrella_map[observation[0]]
    predicted_weather=weather_map[prediction.item()]
    print(f"Observation: {umbrella_status}, Predicted Weather: {predicted_weather}")


