const axios = require("axios"); // For fetching data
const moment = require("moment"); // For date manipulations
const tf = require("@tensorflow/tfjs-node"); // TensorFlow.js for machine learning

// Helper function to calculate the mean of an array segment
function getMean(array, low, high, nullValue = 999) {
  let avg = NaN;
  while (isNaN(avg)) {
    const segment = array.slice(low, high + 1).filter((value) => {
      return parseInt(value) !== nullValue && !isNaN(parseFloat(value));
    });
    if (segment.length > 0) {
      avg = segment.reduce((sum, value) => sum + parseFloat(value), 0) / segment.length;
    } else {
      avg = NaN;
    }

    if (isNaN(avg)) {
      low -= 1;
      high += 1;
    }

    if (high - low > 24) {
      avg = 1e-5;
    }
  }
  return parseFloat(avg.toFixed(5));
}

// Function to fetch solar wind data
async function readOmni(time = "2015-04-04T16:32:00", duration = 6) {
  const year = time.substring(0, 4);
  const url = `https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_${year}.dat`;

  try {
    const response = await axios.get(url, { timeout: 30000 });
    const data = response.data.split("\n").map((line) => line.trim().split(/\s+/));
    const timeMoment = moment(time);
    const doy = timeMoment.diff(moment(`${timeMoment.year()}-01-01`), "days");
    let idx = doy * 24 + timeMoment.hour();

    const wind = {
      Bz: getMean(data.map((row) => row[14]), idx, idx + duration),
      Ratio: getMean(data.map((row) => row[27]), idx, idx + duration, 9),
      V: getMean(data.map((row) => row[24]), idx, idx + duration, 9999),
      Lat: getMean(data.map((row) => row[26]), idx, idx + duration),
      P: getMean(data.map((row) => row[28]), idx, idx + duration, 99),
      Lon: getMean(data.map((row) => row[25]), idx, idx + duration),
      Bx: getMean(data.map((row) => row[12]), idx, idx + duration),
      T: getMean(data.map((row) => row[22]), idx, idx + duration, 9999999),
    };

    return wind;
  } catch (error) {
    console.error("Error fetching solar wind data:", error.message);
    return {};
  }
}

// Function to prepare the input for the SVM model
function getSvmInput(info, features) {
  const x = [];

  // Map for CME and solar wind features
  const featureMap = {
    "CME Average Speed": info["Speed"],
    "CME Final Speed": info["Speed_final"],
    "CME Angular Width": info["Width"],
    "CME Mass": info["Mass"],
    "CME Position Angle": info["PA"],
    "Solar Wind Bz": info.Wind["Bz"],
    "Solar Wind Speed": info.Wind["V"],
    "Solar Wind Temperature": info.Wind["T"],
    "Solar Wind Pressure": info.Wind["P"],
    "Solar Wind Longitude": info.Wind["Lon"],
    "Solar Wind He Proton Ratio": info.Wind["Ratio"],
    "Solar Wind Bx": info.Wind["Bx"],
  };

  // Ensure all features are included
  features.forEach((feature) => {
    if (featureMap[feature] !== undefined) {
      x.push(parseFloat(featureMap[feature]));
    } else {
      x.push(0.0); // Default value for missing features
    }
  });

  // Ensure the array has the correct shape
  return tf.tensor2d([x]);
}

// Function to create and train a simple neural network model
async function createAndTrainModel(features, labels) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: "relu", inputShape: [features.shape[1]] }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1 })); // Single output

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  console.log("Training model...");
  await model.fit(features, labels, { epochs: 50, batchSize: 32, validationSplit: 0.2 });
  console.log("Model training complete.");

  return model;
}

// Main function
(async () => {
  const featuresList = [
    "CME Average Speed",
    "CME Final Speed",
    "CME Angular Width",
    "CME Mass",
    "Solar Wind Bz",
    "Solar Wind Speed",
    "Solar Wind Temperature",
    "Solar Wind Pressure",
    "Solar Wind Longitude",
    "Solar Wind He Proton Ratio",
    "Solar Wind Bx",
    "CME Position Angle",
  ];

  const time = "2015-12-28T12:12:00"; // CME Onset time
  const width = 360; // Angular width
  const speed = 1212; // Speed in km/s
  const finalSpeed = 1243; // Final speed in km/s
  const mass = 1.9e16; // CME mass
  const mpa = 163; // Position angle

  const wind = await readOmni(time, 6);
  const info = { Speed: speed, Speed_final: finalSpeed, Width: width, Mass: mass, PA: mpa, Wind: wind };

  // Prepare input
  const xInput = getSvmInput(info, featuresList);

  // Generate synthetic training data
  const syntheticFeatures = tf.randomNormal([100, featuresList.length]);
  const syntheticLabels = tf.randomNormal([100, 1]);

  // Train a new model
  const model = await createAndTrainModel(syntheticFeatures, syntheticLabels);

  // Make a prediction
  const travelTimeTensor = model.predict(xInput);
  const travelTime = travelTimeTensor.dataSync()[0];

  // Calculate the predicted arrival time
  const arrivalTime = moment(time).add(travelTime, "hours").format("YYYY-MM-DDTHH:mm:ss");

  console.log(`CME with onset time ${time} UT`);
  console.log(`Will hit the Earth at ${arrivalTime} UT`);
  console.log(`With a transit time of ${travelTime.toFixed(2)} hours`);
})();
