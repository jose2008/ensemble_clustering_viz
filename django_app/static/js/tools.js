

function onAddModel( )
{


	var comboboxValue = document.getElementById('selectBox').value;
	//begin();
	console.log("new model......");
	console.log(comboboxValue);
	if(comboboxValue == 1){
		var container1 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'kmean_model' + APPLICATION_DATA['modelContainers'].length , 'Table':'kmeansTable' + APPLICATION_DATA['modelContainers'].length, 'ensamble':0} );
		drawLeftPanelContainer( container1 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container1);
	}


	if(comboboxValue == 2){
		var container2 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'birch_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'birchTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container2 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container2);
	}


	if(comboboxValue == 3){
		var container3 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'som_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'somTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container3 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container3);
	}


	if(comboboxValue == 4){
		var container4 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'dbscan_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'dbscanTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container4 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container4);
	}


	if(comboboxValue == 5){
		var container5 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'agglomerative_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'agglomerativeTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container5 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container5);
	}

	if(comboboxValue == 6){
		var container6 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'kmean_model_clasic'+ APPLICATION_DATA['modelContainers'].length, 'Table':'kmean_model_clasicTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container6 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container6);
	}

	if(comboboxValue == 7){
		var container7 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'MiniBatchKMeans_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'MiniBatchKMeans_modelTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container7 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container7);
	}

	if(comboboxValue == 8){
		var container8 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'kmedoids_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'kmedoids_modelTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container8 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container8);
	}




	var numberOfSides = APPLICATION_DATA['modelContainers'].length,
    size = 100,
    Xcenter = 200,
    Ycenter = 120;

    var canvas = document.querySelector("#myCanvas");
	var cxt = canvas.getContext("2d");

	cxt.canvas.width  = window.innerWidth;
  	cxt.canvas.height = window.innerHeight;
  	console.log("width.........................2");
  	console.log(window.innerWidth);
    //canvas = document.createElement( "CANVAS" );
    //cxt = canvas.getContext( "2d" );

    //cxt.canvas.width  = window.innerWidth;
  	//cxt.canvas.height = window.innerHeight;

  	cxt.clearRect( 0, 0, canvas.width, canvas.height );


	cxt.beginPath();
	cxt.moveTo (Xcenter +  size * Math.cos(0), Ycenter +  size *  Math.sin(0));          

	if(numberOfSides == 1){
		cxt.arc(Xcenter + size * Math.cos(1 * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(1 * 2 * Math.PI / numberOfSides), 5, 0,Math.PI*2);
		cxt.fill();
	}

	if(numberOfSides ==2){
		x_med = ( Xcenter + size * Math.cos(1 * 2 * Math.PI / numberOfSides) + Xcenter + size * Math.cos(2 * 2 * Math.PI / numberOfSides)  )/2
		y_med = ( Ycenter + size * Math.sin(1 * 2 * Math.PI / numberOfSides) + Ycenter + size * Math.sin(2 * 2 * Math.PI / numberOfSides) )/2
		cxt.arc(x_med, y_med, 5, 0,Math.PI*2);
		cxt.fill();	
	}


	for (var i = 1; i <= numberOfSides;i += 1) {
	    cxt.lineTo (Xcenter + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides));
	    cxt.font = "12px Georgia";
		cxt.fillStyle = 'blue'
	    cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
		//APPLICATION_DATA['vertex'].push( [ Xcenter + size * Math.cos(i * 2 * Math.PI / numberOfSides) , Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides) ]   );
	}

	

	cxt.strokeStyle = "#000000";
	cxt.lineWidth = 1;
	cxt.stroke();


	if(numberOfSides >2){
		cxt.beginPath();
		cxt.arc(Xcenter, Ycenter, 5, 0,Math.PI*2);
		cxt.fill();	
	}

	//cxt.stroke();

	document.getElementById('selectBox').value = 0;

	//APPLICATION_DATA['modelContainers'].push(container1);
// ....d. .ds.ad.as 
}



