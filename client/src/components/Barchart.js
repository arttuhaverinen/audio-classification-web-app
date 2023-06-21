import react from "react";
import { Bar } from "react-chartjs-2";
import { Line } from "react-chartjs-2";
import { defaults } from "react-chartjs-2";
import { Pie } from "react-chartjs-2";

const Barchart = ({ data, showCharts, chartLabel, classes }) => {
  //console.log("bar ", chartLabel);
  //console.log(Object.keys(data));
  const labels = Object.keys(data);
  const values = Object.values(data);
  const summedValues = values.reduce((a, b) => a + b);
  //console.log(summedValues);
  //console.log("labels ", labels);
  //console.log("values ", values);
  //console.log("summed values ", summedValues);

  const converToPercentage = (arr, sum) => {
    let newArr = [];
    let divider = 100 / sum;
    arr.forEach((element) => {
      newArr.push(Math.round(element * divider));
    });
    return newArr;
  };

  const percentageValues = converToPercentage(values, summedValues);
  //console.log("percentage values ", percentageValues);
  return showCharts ? (
    classes == "emotion" ? (
      <div className="div-chart">
        <Bar
          datasetIdKey="id"
          options={{
            scales: {
              x: {
                ticks: {
                  color: "black",
                  font: { size: 15 },
                },
              },
              y: {
                ticks: {
                  color: "black",
                  font: { size: 15 },
                },
                min: 0,
                max: 100,
                stepSize: 10,
              },
            },
          }}
          data={{
            labels: labels,

            datasets: [
              {
                label: chartLabel,
                data: percentageValues,
                backgroundColor: [
                  "rgba(231,34,34,0.7)",
                  "rgba(154,205,50,0.7)",
                  "rgba(47,35,35,0.7)",
                  "rgba(233,255,57,0.7)",
                  "rgba(193,177,157,0.7)",
                  "rgba(42,59,144,0.7)",
                ],
                borderColor: [
                  /*"rgb(255, 99, 132)",
                  "rgb(255, 159, 64)",
                  "rgb(255, 205, 86)",
                  "rgb(75, 192, 192)", */
                  "rgba(0,0,0,0.5)",
                ],
                borderWidth: 1,
              },
            ],
          }}
        />
      </div>
    ) : (
      <Bar
        datasetIdKey="id"
        options={{
          legend: {
            labels: {
              fontColor: "blue",
              fontSize: 30,
            },
          },
          scales: {
            x: {
              ticks: {
                color: "black",
                font: { size: 15 },
              },
            },
            y: {
              ticks: {
                color: "black",
                font: { size: 15 },
              },
              min: 0,
              max: 100,
              stepSize: 10,
            },
          },
        }}
        data={{
          labels: labels,
          datasets: [
            {
              label: chartLabel,
              data: percentageValues,
              backgroundColor: ["rgb(255, 204, 203)", "rgb(215,242,243)"],
              borderColor: ["rgba(0,0,0,0.25)"],
              borderWidth: 1,
            },
          ],
        }}
      />
    )
  ) : (
    <div className="card-numbers">
      {Object.keys(data).map((key, val) => (
        <li key={key}>
          {key} : {percentageValues[val]}{" "}
        </li>
      ))}
    </div>
  );
};

export default Barchart;
