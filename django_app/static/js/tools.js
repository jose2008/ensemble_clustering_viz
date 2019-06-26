

function onAddModel( )
{
	var comboboxValue = document.getElementById('selectBox').value;
	//begin();
	if(comboboxValue == 1){
		var container1 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'kmean_model' + APPLICATION_DATA['modelContainers'].length , 'Table':'kmeansTable' + APPLICATION_DATA['modelContainers'].length, 'ensamble':0} );
		drawLeftPanelContainer( container1 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container1);
		APPLICATION_DATA_copy['modelContainers'].push(container1);
	}


	if(comboboxValue == 2){
		var container2 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'birch_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'birchTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container2 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container2);
		APPLICATION_DATA_copy['modelContainers'].push(container2);
	}


	if(comboboxValue == 3){
		var container3 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'som_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'somTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container3 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container3);
		APPLICATION_DATA_copy['modelContainers'].push(container3);
	}


	if(comboboxValue == 4){
		var container4 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'dbscan_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'dbscanTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container4 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container4);
		APPLICATION_DATA_copy['modelContainers'].push(container4);
	}


	if(comboboxValue == 5){
		var container5 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'agglomerative_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'agglomerativeTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container5 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container5);
		APPLICATION_DATA_copy['modelContainers'].push(container5);
	}

	if(comboboxValue == 6){
		var container6 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'kmean_model_clasic'+ APPLICATION_DATA['modelContainers'].length, 'Table':'kmean_model_clasicTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container6 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container6);
		APPLICATION_DATA_copy['modelContainers'].push(container6);
	}

	if(comboboxValue == 7){
		var container7 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'MiniBatchKMeans_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'MiniBatchKMeans_modelTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container7 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container7);
		APPLICATION_DATA_copy['modelContainers'].push(container7);
	}

	if(comboboxValue == 8){
		var container8 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'kmedoids_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'kmedoids_modelTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container8 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container8);
		APPLICATION_DATA_copy['modelContainers'].push(container8);
	}

	if(comboboxValue == 9){
		var container9 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'spectral_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'spectral_modelTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container9 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container9);
		APPLICATION_DATA_copy['modelContainers'].push(container9);
	}

	if(comboboxValue == 10){
		var container10 = createModel( { 'divParentId' : "leftPanelDiv", 'modelType' : 'gmm_model'+ APPLICATION_DATA['modelContainers'].length, 'Table':'gmm_modelTable'+ APPLICATION_DATA['modelContainers'].length, 'ensamble':0 } );
		drawLeftPanelContainer( container10 ,  { 'dataScatter' : [] , 'dataMatrix':[] , 'dataOption':'#slider-container', 'metric1': 0, 'metric2': 0,'metric3': 0} );
		APPLICATION_DATA['modelContainers'].push(container10);
		APPLICATION_DATA_copy['modelContainers'].push(container10);
	}




	var numberOfSides = APPLICATION_DATA['modelContainers'].length,
    size = 60,

    Xcenter = (document.getElementById('tcanvas').clientWidth)/4;
   	Ycenter = (document.getElementById('tcanvas').clientHeight)/2;

   	Xcenter2 = (document.getElementById('tcanvas2').clientWidth)/4;
   	Ycenter2 = (document.getElementById('tcanvas2').clientHeight)/2;

	Xcenter3 = (document.getElementById('tcanvas3').clientWidth)/4;
   	Ycenter3 = (document.getElementById('tcanvas3').clientHeight)/2;

    var canvas = document.querySelector("#myCanvas");
	var cxt = canvas.getContext("2d");
	cxt.canvas.width  = window.innerWidth;
  	cxt.canvas.height = window.innerHeight;
  	cxt.clearRect( 0, 0, canvas.width, canvas.height );

	var canvas2 = document.querySelector("#myCanvas2");
	var cxt2 = canvas2.getContext("2d");
	cxt2.canvas.width  = window.innerWidth;
	cxt2.canvas.height = window.innerHeight;
	cxt2.clearRect( 0, 0, canvas2.width, canvas2.height );

	var canvas3 = document.querySelector("#myCanvas3");
	var cxt3 = canvas3.getContext("2d");
	cxt3.canvas.width  = window.innerWidth;
	cxt3.canvas.height = window.innerHeight;
	cxt3.clearRect( 0, 0, canvas3.width, canvas3.height );



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
	    cxt.font = "10px Georgia";
		cxt.fillStyle = 'blue'
	   if(i==1){
	   		 cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter +3+ size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	   }
	   if(i==2){
	   		if(numberOfSides==4){
	   			cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter -15+ size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter-5 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	   		}
	   		else{
	   			cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter -29+ size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	   		}

	   		 
	   }
	   if(i==3){
	   		 cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter-17 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter-5 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	   }
	   if(i==4){
	   		if(numberOfSides==4){
	   			cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter +3+ size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	   		}
	   		else{
	   			cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter -40+ size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	   		}

	   		 
	   }
	   if (i==5) {
	   		 cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter+2 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	   }
	   if(i==6){
	   		 cxt.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter +1+ size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter -5+ size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	   }
	    //cxt.fillText("M"+i,Xcenter + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides));

	}
	cxt.strokeStyle = "#000000";
	cxt.lineWidth = 1;
	cxt.stroke();
	if(numberOfSides >2){
		cxt.beginPath();
		cxt.arc(Xcenter, Ycenter, 5, 0,Math.PI*2);
		cxt.fill();	
	}



	//another polygon
	cxt2.beginPath();
	cxt2.moveTo (Xcenter2 +  size * Math.cos(0), Ycenter2 +  size *  Math.sin(0));          

	if(numberOfSides == 1){
		cxt2.arc(Xcenter2 + size * Math.cos(1 * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(1 * 2 * Math.PI / numberOfSides), 5, 0,Math.PI*2);
		cxt2.fill();
	}

	if(numberOfSides ==2){
		x_med2 = ( Xcenter2 + size * Math.cos(1 * 2 * Math.PI / numberOfSides) + Xcenter2 + size * Math.cos(2 * 2 * Math.PI / numberOfSides)  )/2
		y_med2 = ( Ycenter2 + size * Math.sin(1 * 2 * Math.PI / numberOfSides) + Ycenter2 + size * Math.sin(2 * 2 * Math.PI / numberOfSides) )/2
		cxt2.arc(x_med2, y_med2, 5, 0,Math.PI*2);
		cxt2.fill();	
	}

	for (var i = 1; i <= numberOfSides;i += 1) {
	    cxt2.lineTo (Xcenter + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter + size * Math.sin(i * 2 * Math.PI / numberOfSides));
	    cxt2.font = "10px Georgia";
		cxt2.fillStyle = 'blue'
	    if(i==1){
	    	cxt2.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter2+3 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    }
	    if(i==2){
	    	if(numberOfSides==4){
	    		cxt2.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter2-15 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2-5 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    	}else{
	    		cxt2.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter2-29 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    	}
	    	
	    }
	    if(i==3){
	    	cxt2.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter2-17 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2-5 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    }
	    if(i==4){
	    	if(numberOfSides==4){
	    		cxt2.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter2+3 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    	}else{
	    		cxt2.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter2-40 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    	}
	    	
	    }
	    if(i==5){
	    	cxt2.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter2+2 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    }
	    if(i==6){
	    	cxt2.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter2+1 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2-5 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    }
		//cxt2.fillText("M"+i,Xcenter2 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides));
	}
	cxt2.strokeStyle = "#000000";
	cxt2.lineWidth = 1;
	cxt2.stroke();
	if(numberOfSides >2){
		cxt2.beginPath();
		cxt2.arc(Xcenter2, Ycenter2, 5, 0,Math.PI*2);
		cxt2.fill();	
	}


	//another polygon 3
	cxt3.beginPath();
	cxt3.moveTo (Xcenter3 +  size * Math.cos(0), Ycenter3 +  size *  Math.sin(0));          

	if(numberOfSides == 1){
		cxt3.arc(Xcenter3 + size * Math.cos(1 * 2 * Math.PI / numberOfSides), Ycenter3 + size * Math.sin(1 * 2 * Math.PI / numberOfSides), 5, 0,Math.PI*2);
		cxt3.fill();
	}

	if(numberOfSides ==2){
		x_med3 = ( Xcenter3 + size * Math.cos(1 * 2 * Math.PI / numberOfSides) + Xcenter3 + size * Math.cos(2 * 2 * Math.PI / numberOfSides)  )/2
		y_med3 = ( Ycenter3 + size * Math.sin(1 * 2 * Math.PI / numberOfSides) + Ycenter3 + size * Math.sin(2 * 2 * Math.PI / numberOfSides) )/2
		cxt3.arc(x_med3, y_med3, 5, 0,Math.PI*2);
		cxt3.fill();	
	}

	for (var i = 1; i <= numberOfSides;i += 1) {
	    cxt3.lineTo (Xcenter3 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter3 + size * Math.sin(i * 2 * Math.PI / numberOfSides));
	    cxt3.font = "10px Georgia";
		cxt3.fillStyle = 'blue'
	    if(i==1){
	    	cxt3.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter3+3 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    }
	    if(i==2){
	    	if(numberOfSides==4){
	    		cxt3.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter3-15 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2-5 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
			}
			else{
				cxt3.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter3-29 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
			}
	    	
	    }
	    if(i==3){
	    	cxt3.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter3-17 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2-5 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    }
	    if(i==4){
	    	if(numberOfSides==4){
	    		cxt3.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter3+3 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
			}
			else{
				cxt3.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter3-40 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
			}
	    	
	    }
	    if(i==5){
	    	cxt3.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter3+2 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    }
	    if(i==6){
	    	cxt3.fillText(APPLICATION_DATA['modelContainers'][i-1].name.substring(0, APPLICATION_DATA['modelContainers'][i-1].name.length-7)
	    	,Xcenter3+1 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter2-5 + size * Math.sin(i * 2 * Math.PI / numberOfSides)   ) 
	    }
		//cxt3.fillText("M"+i,Xcenter3 + size * Math.cos(i * 2 * Math.PI / numberOfSides), Ycenter3 + size * Math.sin(i * 2 * Math.PI / numberOfSides));
	}
	cxt3.strokeStyle = "#000000";
	cxt3.lineWidth = 1;
	cxt3.stroke();
	if(numberOfSides >2){
		cxt3.beginPath();
		cxt3.arc(Xcenter3, Ycenter3, 5, 0,Math.PI*2);
		cxt3.fill();	
	}



	document.getElementById('selectBox').value = 0;
}



//function to draw......

function inside(point, vs) {
    var x = point[0], y = point[1];
    var inside = false;
    for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
        var xi = vs[i][0], yi = vs[i][1];
        var xj = vs[j][0], yj = vs[j][1];

        var intersect = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
};

function isOnLine(x, y, endx, endy, px, py) {
    var f = function(somex) { return (endy - y) / (endx - x) * (somex - x) + y; };
    return Math.abs(f(px) - py) < 200 // tolerance, rounding errors
        && px >= x && px <= endx;      // are they also on this segment?
}


function triangle_area(x1, y1, x2, y2, x3, y3){
    return Math.abs(0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))
}


function gaussian(std, x){
	return    (1/(std*std*Math.PI))*Math.pow(Math.E, -Math.pow(x,2)/(std*std)  );

};


function distance(l1,l2){
	return Math.sqrt(  (l1[0]-l2[0])*(l1[0]-l2[0])  +   (l1[1]-l2[1])*(l1[1]-l2[1])  )/10;
}

function float2color( percentage ) {
    var color_part_dec = 255 * percentage;
    var color_part_hex = Number(parseInt( color_part_dec , 10)).toString(16);
    return "#" + color_part_hex + color_part_hex + color_part_hex;
}





 getGradientColorGreen = function(start_color, end_color, percent) {
   // strip the leading # if it's there
   start_color = start_color.replace(/^\s*#|\s*$/g, '');
   end_color = end_color.replace(/^\s*#|\s*$/g, '');

   // convert 3 char codes --> 6, e.g. `E0F` --> `EE00FF`
   if(start_color.length == 3){
     start_color = start_color.replace(/(.)/g, '$1$1');
   }

   if(end_color.length == 3){
     end_color = end_color.replace(/(.)/g, '$1$1');
   }

   // get colors
   var start_red = parseInt(start_color.substr(0, 2), 16),
       start_green = parseInt(start_color.substr(2, 2), 16),
       start_blue = parseInt(start_color.substr(4, 2), 16);

   var end_red = parseInt(end_color.substr(0, 2), 16),
       end_green = parseInt(end_color.substr(2, 2), 16),
       end_blue = parseInt(end_color.substr(4, 2), 16);

   // calculate new color
   var diff_red = end_red - start_red;
   var diff_green = end_green - start_green;
   var diff_blue = end_blue - start_blue;

   diff_red = ( (diff_red * percent) + start_red ).toString(16).split('.')[0];
   diff_green = ( (diff_green * percent) + start_green ).toString(16).split('.')[0];
   diff_blue = ( (diff_blue * percent) + start_blue ).toString(16).split('.')[0];

   // ensure 2 digits by color
   if( diff_red.length == 1 ) diff_red = '0' + diff_red
   if( diff_green.length == 1 ) diff_green = '0' + diff_green
   if( diff_blue.length == 1 ) diff_blue = '0' + diff_blue

   return '#' + diff_red + diff_green + diff_blue;
 };

function arrayMin(arr) {
  //if(arr) return;
  return arr.reduce(function (p, v) {
  return ( p < v ? p : v );
  });
}

function arrayMax(arr) {
  //if(arr) return ;
  return arr.reduce(function (p, v) {
  return ( p > v ? p : v );
  });
}

function toggleIcon(e) {
        $(e.target)
            .prev('.panel-heading')
            .find(".more-less")
            .toggleClass('glyphicon-plus glyphicon-minus');
    }
    $('.panel-group').on('hidden.bs.collapse', toggleIcon);
    $('.panel-group').on('shown.bs.collapse', toggleIcon);




function intersection_destructive(a, b)
{
  var result = [];
  while( a.length > 0 && b.length > 0 )
  {  
     if      (a[0] < b[0] ){ a.shift(); }
     else if (a[0] > b[0] ){ b.shift(); }
     else /* they're equal */
     {
       result.push(a.shift());
       b.shift();
     }
  }

  return result;
}