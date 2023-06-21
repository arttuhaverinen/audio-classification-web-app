const Settings = ({
  isCheckedDensenet,
  setIsCheckedDensenet,
  isCheckedResnet,
  setIsCheckedResnet,
  isCheckedAlexnet,
  setIsCheckedAlexnet,
  isCheckedMobilenet,
  setIsCheckedMobilenet,
  showCharts,
  setShowCharts,
  showLargeCharts,
  setShowLargeCharts,
  csv,
  setCsv,
  CSVLink,
}) => {
  return (
    <div className="div-settings">
      <h2>Settings</h2>
      <hr />
      <form action="">
        <div>
          <input
            checked={isCheckedDensenet}
            onChange={() => setIsCheckedDensenet(!isCheckedDensenet)}
            type="checkbox"
            id="densenet"
          />
          <label for="densenet"> DenseNet-121 </label> <br />
        </div>
        <div>
          <input
            checked={isCheckedResnet}
            onChange={() => setIsCheckedResnet(!isCheckedResnet)}
            type="checkbox"
            id="resnet"
          />
          <label for="resnet"> ResNet50</label> <br />
        </div>
        <div>
          <input
            checked={isCheckedAlexnet}
            onChange={() => setIsCheckedAlexnet(!isCheckedAlexnet)}
            type="checkbox"
            id="alexnet"
          />
          <label for="alexnet"> AlexNet</label> <br />
        </div>
        <div>
          <input
            checked={isCheckedMobilenet}
            onChange={() => setIsCheckedMobilenet(!isCheckedMobilenet)}
            type="checkbox"
            id="mobilenet"
          />
          <label for="mobilenet"> MobileNetv3</label> <br />
        </div>
        <hr />
        <div>
          <input
            checked={showCharts}
            onChange={() => setShowCharts(!showCharts)}
            type="checkbox"
            id="showcharts"
          />
          <label for="toggleGraph"> Show graphs </label> <br />
        </div>
        <div>
          <input
            checked={showLargeCharts}
            onChange={() => setShowLargeCharts(!showLargeCharts)}
            type="checkbox"
            id="showLargeCharts"
          />
          <label for="showLargeCharts"> Large charts </label> <br />
        </div>
        <hr />
        <div>
          {csv ? (
            <CSVLink
              enclosingCharacter={``}
              className="csv-download"
              filename="csvfile"
              target="_blank"
              data={csv}
            >
              Download csv
            </CSVLink>
          ) : null}
        </div>
      </form>
    </div>
  );
};

export default Settings;
