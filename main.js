const axios = require('axios');
const moment = require('moment');
const fs = require('fs');
const path = require('path');

// Helper function to calculate the mean of an array slice
function getMean(array, low, high, nullValue = 999) {
  let flag = 0;
  if (low > high) {
    flag = 1;
    [low, high] = [high, low];
  }

  let avg = NaN;
  while (isNaN(avg)) {
    const dust = array.slice(low, high + 1).filter(value => parseInt(value) !== nullValue && !isNaN(parseFloat(value)));

    if (dust.length !== 0) {
      avg = dust.reduce((sum, value) => sum + parseFloat(value), 0) / dust.length;
    } else {
      avg = NaN;
    }

    if (isNaN(avg)) {
      if (flag === 0) {
        low -= 1;
      } else {
        high += 1;
      }
    }
  }

  if (high - low > 24) {
    avg = 1e-5;
  }

  return avg.toFixed(5);
}

// Function to fetch solar wind data
async function readOmni(time = '2015-04-04T16:32:00', duration = 6) {
  const year = time.substring(0, 4);
  const url = `https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_${year}.dat`;

  try {
    const response = await axios.get(url);
    const data = response.data.split("\n").map(line => line.trim().split(/\s+/));

    const dateTime = moment(time);
    const doy = dateTime.diff(moment(`${dateTime.year()}-01-01`), 'days');
    let idx = doy * 24 + dateTime.hour();

    const wind = {
      Bz: getMean(data.map(row => row[14]), idx, idx + duration),
      Ratio: getMean(data.map(row => row[27]), idx, idx + duration, 9),
      V: getMean(data.map(row => row[24]), idx, idx + duration, 9999),
      Lat: getMean(data.map(row => row[26]), idx, idx + duration),
      P: getMean(data.map(row => row[28]), idx, idx + duration, 99),
      Lon: getMean(data.map(row => row[25]), idx, idx + duration),
      Bx: getMean(data.map(row => row[12]), idx, idx + duration),
      T: getMean(data.map(row => row[22]), idx, idx + duration, 9999999)
    };

    return wind;
  } catch (error) {
    console.error('Error fetching solar wind data:', error);
    return {};
  }
}

// Function to prepare input for the SVM engine
function getSvmInput(info, features) {
  const sortedFeatures = features.sort();
  const x = [];

  const featureMap = {
    "CME Acceleration": info['Acceleration'],
    "CME Angular Width": info['Width'],
    "CME Average Speed": info['Speed'],
    "CME Final Speed": info['Speed_final'],
    "CME Mass": info['Mass'],
    "CME Position Angle": info['PA'],
    "CME Source Region Latitude": info['Lat'],
    "CME Source Region Longitude": info['Lon'],
    "CME Speed at 20 Rs": info['Speed_20'],
  };

  sortedFeatures.forEach(feature => {
    if (featureMap[feature] !== undefined) {
      x.push(parseFloat(featureMap[feature]));
    } else if (info.Wind[feature] !== undefined) {
      x.push(parseFloat(info.Wind[feature]));
    }
  });

  return [x];
}

// Function to load the SVM engine from the file
function loadEngine(filePath) {
  const data = fs.readFileSync(filePath);
  return JSON.parse(data); // Assuming the engine is serialized as JSON for simplicity
}

// Function to calculate the arrival time and prediction error
async function main() {
  // Feature set for the model
  const features = [
    'CME Average Speed',
    'CME Final Speed',
    'CME Angular Width',
    'CME Mass',
    'Solar Wind Bz',
    'Solar Wind Speed',
    'Solar Wind Temperature',
    'Solar Wind Pressure',
    'Solar Wind Longitude',
    'Solar Wind He Proton Ratio',
    'Solar Wind Bx',
    'CME Position Angle'
  ];

  // Parameters
  const time = '2015-12-28T12:12:00'; // CME Onset time in LASCO C2
  const width = 360; // Angular width, degree, set as 360 if it is halo
  const speed = 1212; // Linear speed in LASCO FOV, km/s
  const finalSpeed = 1243; // Second order final speed leaving LASCO FOV, km/s
  const mass = 1.9e16; // Estimated mass of the CME
  const mpa = 163; // Position angle corresponding to the fastest front
  const actual = '2015-12-31T00:02:00'; // Actual arrival time

  // Fetch solar wind parameters
  const wind = await readOmni(time);

  const info = {
    CME: time,
    Speed: speed,
    Speed_final: finalSpeed,
    Width: width,
    Mass: mass,
    PA: mpa,
    Wind: wind
  };

  // Get input for the SVM engine
  const xInput = getSvmInput(info, features);

  // Load the SVM engine (replace 'engine.json' with your engine file)
  const engineFile = './engine.json';
  const engine = loadEngine(engineFile);

  // Normalize input (assuming engine has a 'scaler' object with 'transform' method)
  const normalizedInput = engine.scaler.transform(xInput);

  // Do Prediction
  const travel = engine.clf.predict(normalizedInput)[0];

  // Calculate the arrival time
  const arriveTime = moment(time).add(travel, 'hours').format('YYYY-MM-DDTHH:mm:ss');

  // Show Results
  console.log(`CME with onset time ${time} UT`);
  console.log(`Will hit the Earth at ${arriveTime} UT`);
  console.log(`With a transit time of ${travel} hours`);

  if (actual) {
    const diff = moment(arriveTime).diff(moment(actual), 'hours');
    console.log(`The actual arrival time is ${actual} UT`);
    console.log(`The prediction error is ${diff} hours`);
  }
}

// Run the main function
main().catch(console.error);
