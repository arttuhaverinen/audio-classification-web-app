// Create CSV function. Works but needs some work...

const handleCsvCreation = (emotionsData, setCsv) => {
  let highest = -Infinity;
  let correctClass = null;
  let csvArr = ["name"];
  console.log("emotions loaded");
  if (emotionsData) {
    emotionsData.forEach((arr) => {
      arr.forEach((element) => {
        console.log(element);
        for (const val in element) {
          console.log(`${val}, ${element[val]}`);
          if (element[val] > highest) {
            highest = element[val];
            correctClass = val;
          }
        }
        highest = -Infinity;
        console.log("correctclass ", correctClass);
        csvArr.push(correctClass);
      });
    });
    console.log("csvarr", csvArr);
    let headers = [
      "File",
      "DenseNet121_Gender",
      //"Resnet50_Gender",
      //"AlexNet_Gender",
      "DenseNet121_Emotion",
      //"Resnet50_Emotion",
      //"AlexNet_Emotion",
      "MobileNetv3_Gender",
      "MobileNetv3_Emotion",
    ];
    setCsv([[headers.slice()], csvArr.slice()]);
  }
};

export default handleCsvCreation;
