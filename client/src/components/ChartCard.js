import react, { useEffect, useState } from "react";
import Barchart from "./Barchart";

const ChartCard = ({
  header,
  emotionsData,
  ischecked,
  showCharts,
  showLargeCharts,
  chartLabel,
  classes,
}) => {
  const [showPercentages, setShowPercentages] = useState(true);
  const [predictedClass, setPredictedClass] = useState(false);
  const [predictedGender, setPredictedGender] = useState(false);
  const [predictedEmotion, setPredictedEmotion] = useState(false);

  useEffect(() => {
    let highest = -Infinity;
    let correctClass = null;
    //console.log("emotions loaded");
    if (emotionsData) {
      for (const val in emotionsData) {
        //console.log(`${val}, ${emotionsData[val]}`);
        if (emotionsData[val] > highest) {
          highest = emotionsData[val];
          correctClass = val;
        }
      }
      setPredictedClass(correctClass);
    }
  });

  const handleShowPercentages = (e) => {
    if (!emotionsData) return;
    else if (showPercentages) setShowPercentages(false);
    else if (showPercentages == false) setShowPercentages(true);
  };
  //console.log("emotionsdata xx", emotionsData);

  return ischecked ? (
    <div className={showLargeCharts ? "card" : "card-small"}>
      <div className="card-header">
        <h3>{header}</h3>
        {predictedClass ? <h3>Predicted class: {predictedClass} </h3> : null}
        {/*<button onClick={(e) => handleShowPercentages(e)}>Expand/Hide</button>*/}
      </div>
      <div>
        {emotionsData ? (
          <div>
            <Barchart
              data={emotionsData}
              showCharts={showCharts}
              chartLabel={chartLabel}
              classes={classes}
            />
          </div>
        ) : (
          <div></div>
        )}
      </div>
    </div>
  ) : null;
};

export default ChartCard;
