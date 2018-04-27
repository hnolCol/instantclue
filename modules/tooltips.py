




sliceMarksTooltip = {'selection':'Allows for freehand selection of data points in a scatter plot. Right click opens a menu to a) create a annotation column b) subset data or c) to delete selected data.',
					 'filter':'Can receive columns from the source data treeview via drag & drop. Depending on the datatype different a dialog windows opens that allows for filtering. You can also try the new "liver filter" feature by simply clicking on the icon.',
					 'color':'This button has two functionalities: a) Changing the default color palette. b) In several plot types (line, hierarchical clustering, scatter, scatter matrix,PCA ..), column names can be dragged & dropped onto this icon to color code graph items.',
					 'size':'This button has two functionalities: a) Changing the default size of points. b) In several plot types (line, hierarchical clustering, scatter, scatter matrix,PCA ..), column names can be dragged & dropped onto this icon to size encode graph items.',
					 'label':'In a scatter plot you can click this button to label selected data points. You can also use drag & drop of column names onto this button to manually annotate data points in a plot. Notably, if you drag and drop a categorical column that contains only "+" and "-" you will be asked if you want to label only the data points annotated by "+".',
					 'tooltip':'Drag & drop a categorical column onto this button to enable the tooltip. Hovering over data points will display a tooltip containing information of the dropped columns.'}
	

loadButtonTooltip = {'upload':'Load data into Instant Clue. Supported file formats are: Excel, xml, txt, and csv. To add more data frames to the same session, please use the ADD button. Using this button will delete already loaded data.',
					 'addData':'Add data frame to current session. Supported file formats are: Excel, xml, txt, and csv.',
					 'saveSession':'Save a session to be used at a later time point. Session are automatically stored in the Instant Clue directory.',
					 'openSession':'Open a stored session.'}
		
tooltip_information_plotoptions = [
        ["At least two numeric column. Each row is plotted against the column index. Add categorical columns by using the color encoding icon.",
         "Lineplot\n\nInput:"],
        ["At least one numeric column\nMax. Categories for factorplot: 3\nData are represented by a single point showing the confidence interval (0.95) and are connected if they belong to the same group.",
         "Pointplot\n\nInput:"],
        ["At least two numeric columns (maximum 3)\ Categories for factorplot: up to 3 supported. More categories can be added using the color option. Simply drag & drop the desired numerical or categorical column to the color button.\nAdditional categories can be added using drag & drop to the newly created color button.",
         "Scatter\n\nInput:"],
         ["At least two numeric columns\nCategories are not supported.\nTime series option uses the first column as the x-axis and all addition columns are plotted against this value. E.g. the first column should be the time while the following columns are numeric values (measurements)",
         "Time Series\n\nInput:"],
          ["At least two numeric columns\nCategories are not supported.\nColors and Size changes might be added as described for scatter plots.",
         "Scatter Matrix\n\nInput:"],
           ["At least one numeric column\nUnlimtited categories can be added.\nIf categorical columns are present each combination of categories will be used to slice data and to display the density information.",
         "Density plot\n\nInput:"],
            ["At least one numeric column\nMax. Categories for factorplot: 3\nUnlimited when Split categories disabled.\nThe error bars indicate the confidence interval (0.95).\nAdditional Options: Split Categories and split categories in different subplots.",
         "Barplot\n\nInput:"],
        ["At least one numeric column\nMax. Categories for factorplot: 3\nUnlimited when Split categories disabled.\n\nAdditional Options: Split Categories and split categories in different subplots.",
         "Boxplot\n\nInput:"],
         ["At least one numeric column\nMax. Categories for factorplot: 3\nUnlimited when Split categories disabled.\n\nAdditional Options: Split Categories and\nsplit categories in different subplots.\nViolin plot show a boxplot inside as well kernel density information",
         "Violinplot\n\nInput:"],
        ["At least one numeric column\nMax. Categories for factorplot: 3\nUnlimited when Split categories disabled.\n\nAdditional Options: Split Categories and\nsplit categories in different subplots.\nSwarmplots show the raw data points separated by jitter on x-axis.",
         "Swarmplot\n\nInput:"],
        ["Raw datapoints separated by jitter can be added to: Box-, Bar-, and Violinplots. The datapoints cannot be changed in color and size.",
         "Add swarm to plot\n\nInput:"],
         ["At least two numeric column\nCategories are not supported.",
         "Hierachical Clustering\n\nInput:"],
        ["At least two numeric columns.\nA correlation matrix calculates the Pearson correlation coefficient and uses hierachical clustering for interpretation.",
         "Correlation Matrix\n\nInput:"],
        # ["Up to one nuermic column to control size of each node.\nAt least two categorical columns to describe the edges.",
        # "Network\n\nInput:"],
           ["Opens a popup that allows you too change details of your created chart.\nNote that you can also save settings as templates and load them to get the exact same chart configuration.\nFor same coloring it is essential that the categories are sorted in the same way as in the template.",
         "Configure\n\n"]
        ]	
		
			





