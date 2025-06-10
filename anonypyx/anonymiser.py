from anonypyx import generalisation, models, algorithms

class Anonymiser:
    def __init__(self, df, **kwargs):
        '''
        Creates a new Anonymiser instance.  
        Checks whether the given configuration options are valid.

        Parameters
        ----------
            df : pandas.DataFrame
                The pandas DataFrame to anonymise.

        Keyword Parameters
        ------------------
            sensitive_column : str
                The name of the sensitive attribute column in the data frame. Must be set when l-diversity or t-closeness are applied. (default: None)
            feature_columns : array-like
                The names of the quasi-identifier columns in the data frame. Anonymisation only changes these columns. (default: All columns)
            generalisation_strategy : str
                Determines how the quasi-identifiers in the anonymised data are generalised.
                Available options are "human-readable", "machine-readable" and "microaggregation".
                "microaggregation" does not support categorical values.
                (default: "machine-readable")
            k : int
                Parameter k of k-anonymity. (default: 1)
            l : int
                Parameter l of l-diversity. Setting this to None deactivates l-diversity. (default: None)
            t : float
                Parameter t of t-closeness. Setting this to None deactivates t-closeness. (default: None)
            diversity_definition : str
                Instantiation of l-diversity principle to use. Can be either "distinct", "simple", "entropy" or "recursive". (default: "distinct")
            closeness_metric : str
                Distance metric used by t-closeness. Can be either "max distance" or "earth mover's distance". (default: "max distance")
            algorithm : str
                The anonymisation algorithm to use. Can be either "Mondrian" or "MDAV-generic" (supports only k-anonymity). (default: "Mondrian")

        Raises
        ------
            TypeError 
                When a parameter has the wrong type.
            ValueError 
                When a single parameter value or a combination of parameter values are invalid.
        
        '''
        sensitive_attribute = kwargs.get("sensitive_column", None)
        quasi_identifiers = kwargs.get("feature_columns", df.columns)
        anonymity_models = []
        k = kwargs.get("k", 1)
        l = kwargs.get("l", None)
        t = kwargs.get("t", None)
        l_diversity_definition = kwargs.get("diversity_definition", "distinct")
        t_closeness_metric = kwargs.get("closeness_metric", "max distance")
        algorithm = kwargs.get("algorithm", "Mondrian")
        # aggregations = {}
        generalisation_strategy = kwargs.get("generalisation_strategy", "machine-readable")
        generalisation_strategy_type = None

        if generalisation_strategy == 'machine-readable':
            generalisation_strategy_type = generalisation.MachineReadable
        elif generalisation_strategy == 'human-readable':
            generalisation_strategy_type = generalisation.HumanReadable
        elif generalisation_strategy == 'microaggregation':
            generalisation_strategy_type = generalisation.Microaggregation
        else:
            raise TypeError("Unknown generalisation strategy.")

        if type(k) is not int:
            raise TypeError("k must be an integer.")
    
        if (l is not None) and (type(l) is not int):
            raise TypeError("l must be an integer.")
    
        if (t is not None) and (type(t) is not float):
            raise TypeError("t must be a float.")
    
        if type(l_diversity_definition) is not str:
            raise TypeError("diversity_definition must be a string.")
        
        if type(algorithm) is not str:
            raise TypeError("algorithm must be a string")
    
        if k < 1 or k > len(df.index):
            raise ValueError("k must be between 1 and the number of records.")
    
        if (l is not None) and (l < 1):
            raise ValueError("l must be greater than 1.")
    
        # TODO: feasibility check for large l values?
    
        if (t is not None) and (t < 0 or t > 1):
            raise ValueError("t must be between 0 and 1.")
    
        if (sensitive_attribute is not None) and (sensitive_attribute not in df.columns):
            raise ValueError("sensitive_column must be a column name within the given data frame.")
    
        if not all(qi in df.columns for qi in quasi_identifiers):
            raise ValueError("Every feature column in feature_columns must be a column name within the given data frame.")
    
        privacy_models = [models.kAnonymity(k)]
    
        if l is not None:
            if l_diversity_definition == "distinct":
                privacy_models.append(models.DistinctLDiversity(l, sensitive_attribute))
            elif l_diversity_definition == "simple":
                raise NotImplementedError("Simple l-diversity has not been implemented yet.")
            elif l_diversity_definition == "entropy":
                raise NotImplementedError("Entropy l-diversity has not been implemented yet.")
            elif l_diversity_definition == "recursive":
                raise NotImplementedError("Recursive (c,l)-diversity has not been implemented yet.")
            else:
                raise ValueError("diversity_definition does not match any known instantiation of the l-diversity principle.")
    
        if t is not None:
            if t_closeness_metric == "max distance":
                privacy_models.append(models.tCloseness(t, df, sensitive_attribute, models.max_distance_metric))
            elif t_closeness_metric == "earth mover's distance":
                if not all(df[qi].dtype.name == "category" for qi in quasi_identifiers):
                    raise NotImplementedError("Earth mover's distance has not been implemented for numerical attributes yet.")
                privacy_models.append(models.tCloseness(t, df, sensitive_attribute, models.earth_movers_distance_categorical))
    
        if algorithm == "Mondrian":
            self.algorithm = algorithms.Mondrian(privacy_models, quasi_identifiers)
        elif algorithm == "MDAV-generic":
            if l is not None:
                raise ValueError("algorithm 'MDAV-generic' does not support l-diversity.")
            if t is not None:
                raise ValueError("algorithm 'MDAV-generic' does not support t-closeness.")
    
            self.algorithm = algorithms.MDAVGeneric(k, quasi_identifiers)
        self.df = df
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attribute = sensitive_attribute
        self.generalisation_strategy_type = generalisation_strategy_type

    def anonymise(self):
        '''
        Starts the anonymisation algorithm with the options specified by this Anonymiser instance.

        Returns
        -------
            List of anonymised records.
        '''
        partitions = self.algorithm.partition(self.df)
        generalisation = self.generalisation_strategy_type.create_for_data(self.df, self.quasi_identifiers)
        return generalisation.generalise(self.df, partitions)

