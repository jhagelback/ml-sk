
package aml;

import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.*;

/**
 * Naive Bayes classifier using Weka.
 * 
 * @author Johan Hagelbäck (johan.hagelback@lnu.se)
 */
public class TextClassification 
{
    /** Path to dataset */
    private String filename;
    /** Dataset in Weka instances */
    private Instances data;
    /** Weka classifier */
    private Classifier cl;
    
    /**
     * Runs everything
     */
    public static void run()
    {
        TextClassification t = new TextClassification("data/wikipedia_70.arff");
        
        System.out.println("Accuracy (whole dataset):");
        t.evaluateAll();
        
        System.out.println("Accuracy (10-fold CV):");
        t.evaluateCV();
    }
    
    /**
     * Constructor.
     * 
     * @param filename Path to dataset file
     */
    public TextClassification(String filename)
    {
        this.filename = filename;
        readData();
    }
    
    /**
     * Reads data from an ARFF file.
     */
    private void readData()
    {
        try
        {
            //Read raw data (all words are in one string)
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
            Instances raw = source.getDataSet();
            
            //Convert to bag-of-words using the StringToWordVector filter
            StringToWordVector stw = new StringToWordVector(10000);
            stw.setLowerCaseTokens(true);
            stw.setInputFormat(raw);
            
            data = Filter.useFilter(raw, stw);
            /*for (int a = 0; a < data.numAttributes(); a++)
            {
                Attribute attr = data.attribute(a);
                System.out.println(attr.name() + ":" + data.instance(0).value(a));
            }*/
            
            //If StringToWordVector is used, Weka puts the
            //class attribute first (in contrast to the default
            //where class attribute is last)
            data.setClassIndex(0);
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
            System.exit(0);
        }
    }
    
    /**
     * Evaluates the classifier using the whole dataset for both
     * training and testing.
     */
    public void evaluateAll()
    {
        try
        {
            cl = new NaiveBayesMultinomial();
            cl.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(cl, data);
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
            
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
    
    /**
     * Evaluates the classifier using 10-fold cross validation.
     */
    public void evaluateCV()
    {
        try
        {
            cl = new NaiveBayesMultinomial();
            cl.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cl, data, 10, new java.util.Random(1));
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
            
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
