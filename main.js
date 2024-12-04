const tf = require("@tensorflow/tfjs-node");
const axios = require("axios");
const moment = require("moment");

// Fetch and preprocess CME data from the provided URL
async function fetchCMEData() {
  const url = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver/univ2024_08.txt";
  try {
    const response = await axios.get(url);
    const lines = response.data.split("\n").filter((line) => line.trim() !== "");

    const processedData = [];
    const labels = [];

    // Extract relevant fields: Linear Speed, Angular Width, Transit Time (if available)
    for (const line of lines) {
      const fields = line.trim().split(/\s+/); // Split line into fields
      if (fields.length >= 7) {
        const speed = parseFloat(fields[3]); // Linear Speed (km/s)
        const width = parseFloat(fields[5]); // Angular Width (degrees)
        const transitTime = parseFloat(fields[6]); // Transit Time (hours)

        // Validate data before adding to processedData and labels
        if (!isNaN(speed) && !isNaN(width) && !isNaN(transitTime) && transitTime > 0) {
          processedData.push([speed, width]);
          labels.push(transitTime);
        }
      }
    }

    // Check if processedData and labels are not empty
    if (processedData.length === 0 || labels.length === 0) {
      throw new Error("No valid data found in the CME file.");
    }

    console.log("Processed Data:", processedData);
    console.log("Labels:", labels);

    return {
      features: tf.tensor2d(processedData),
      labels: tf.tensor2d(labels, [labels.length, 1]),
    };
  } catch (error) {
    console.error("Error fetching CME data:", error.message);
    return { features: tf.tensor2d([], [0, 2]), labels: tf.tensor2d([], [0, 1]) }; // Return empty tensors if no data
  }
}

// Normalize data
function normalize(tensor) {
  const min = tensor.min();
  const max = tensor.max();
  return tensor.sub(min).div(max.sub(min));
}

// Create and train the model
async function createAndTrainModel(features, labels) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: "relu", inputShape: [features.shape[1]] }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1 })); // Single output

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  console.log("Training model...");
  await model.fit(features, labels, { epochs: 50, batchSize: 32, validationSplit: 0.2 });
  console.log("Model training complete.");
  return model;
}

// Prepare input for prediction
function prepareInput(params) {
  const { speed, width } = params;
  return tf.tensor2d([[speed, width]]);
}

// Main function
(async () => {
  console.log("Fetching and preprocessing CME data...");
  const { features, labels } = await fetchCMEData();

  // Check if we have enough data
  if (features.shape[0] === 0 || labels.shape[0] === 0) {
    console.error("No data available for training. Exiting...");
    return;
  }

  // Normalize the data
  const normalizedFeatures = normalize(features);
  const normalizedLabels = normalize(labels);

  // Train the model
  const model = await createAndTrainModel(normalizedFeatures, normalizedLabels);

  // Example CME parameters for prediction
  const cmeParams = {
    speed: 1212, // Linear speed in km/s
    width: 360,  // Angular width in degrees
  };

  // Prepare the input
  const inputTensor = prepareInput(cmeParams);

  // Make the prediction
  const predictedTransitTime = model.predict(inputTensor).dataSync()[0];

  // Validate and calculate the arrival time
  if (isNaN(predictedTransitTime) || predictedTransitTime <= 0) {
    console.error("Invalid prediction:", predictedTransitTime);
    return;
  }

  const cmeOnsetTime = "2015-12-28T12:12:00";
  const arrivalTime = moment(cmeOnsetTime)
    .add(predictedTransitTime, "hours")
    .format("YYYY-MM-DDTHH:mm:ss");

  console.log(`CME with onset time ${cmeOnsetTime} UT`);
  console.log(`Will hit the Earth at ${arrivalTime} UT`);
  console.log(`With a transit time of ${predictedTransitTime.toFixed(2)} hours`);
})();
