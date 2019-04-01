
var APPLICATION_DATA = {};


function createModel( params )
{
	//var _container = new VizContainer( params['containerDiv'] );
	var _divParentId = params['divParentId'];
	var _divParent = document.getElementById( _divParentId );
	var _divContainer = document.createElement( 'div' );
	_divParent.appendChild( _divContainer );

	var _container;
	if( params['ensamble'] == 0){
		_container = new VizContainer( _divContainer, {'modelType': params['modelType'], 'Table':params['Table']} );
	}else{
		_container = new VizContainerMain( _divContainer, {'modelType': params['modelType'], 'Table':params['Table']} );
	}
	return _container;
};




function drawLeftPanelContainer( modelContainer, params )
{
	modelContainer.draw( params );
}


function drawRightPanelContainer( modelContainer, params )
{
	modelContainer.draw( params );
}



function draw()
{
	for ( var i = 0; i < APPLICATION_DATA['numModels']; i++ )
	{
		drawLeftPanelContainer( APPLICATION_DATA['modelContainers'][i], {} );
	}
};



// function ApplicationModelsPanel()
// {
// 	this.m_modelContainers = [];
// };
// 
// 
// ApplicationModelsPanel.prototype.addModelView = function( params ) 
// {
// 	// create new element for this new model container
// 	var _divContainer = document.createElement( 'div' );
// 
// };
// 
// 
// var APPLICATION_MODEL_PANEL = null;


function begin()
{


	





	// initialize everything here
	// procedural
	APPLICATION_DATA['leftPanelDiv'] 	= document.getElementById( 'leftPanelDiv' );
	APPLICATION_DATA['rightPanelDiv'] 	= document.getElementById( 'rightPanelDiv' );
	APPLICATION_DATA['numModels'] 		= 0;
	APPLICATION_DATA['modelContainers'] = [];
	APPLICATION_DATA['metrics'] = [];
	APPLICATION_DATA['points'] = [];
	APPLICATION_DATA['metric'] ;
	APPLICATION_DATA['vertex'] = [];
	






	// oop
	//APPLICATION_MODEL_PANEL = new ApplicationModelsPanel();
};