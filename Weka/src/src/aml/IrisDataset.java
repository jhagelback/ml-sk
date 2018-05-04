
package aml;

import java.text.DecimalFormat;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.classifiers.bayes.*;
import weka.classifiers.trees.*;
import weka.classifiers.lazy.IBk;
import weka.classifiers.*;

/**
 * Naive Bayes classifier using Weka.
 * 
 * @author Johan Hagelbäck (johan.hagelback@lnu.se)
 */
public class IrisDataset 
{
    /** Path to dataset */
    private String filename;
    /** Dataset in Weka instances */
    private Instances data;
    
    private DecimalFormat df = new DecimalFormat("0.00");
    
    /**
     * Runs everything
     */
    public static void run()
    {
        IrisDataset t = new IrisDataset("data/iris.arff");
        
        System.out.println("\n---- Naive Bayes ----");
        
        System.out.println("Training data:");
        t.evaluateAll(new NaiveBayes());
        
        System.out.println("10-fold CV:");
        t.evaluateCV(new NaiveBayes());
        
        System.out.println("\n---- IBk ----");
        
        System.out.println("Training data:");
        t.evaluateAll(new IBk(3));
        
        System.out.println("10-fold CV:");
        t.evaluateCV(new IBk(3));
        
        System.out.println("\n---- Decision Trees ----");
        
        System.out.println("Training data:");
        t.evaluateAll(new J48());
        
        System.out.println("10-fold CV:");
        t.evaluateCV(new J48());
        
        System.out.println("\n---- Random Forest ----");
        
        System.out.println("Training data:");
        t.evaluateAll(new RandomForest());
        
        System.out.println("10-fold CV:");
        t.evaluateCV(new RandomForest());
    }
    
    /**
     * Constructor.
     * 
     * @param filename Path to dataset file
     */
    public IrisDataset(String filename)
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
            //Read data
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
            data = source.getDataSet();
            //Set class index to last
            data.setClassIndex(data.numAttributes() - 1);
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
     * 
     * @param cl The classifier
     */
    public void evaluateAll(Classifier cl)
    {
        try
        {
            cl.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(cl, data);
            System.out.println("Accuracy: " + df.format(eval.correct()/data.numInstances()*100.0) + "%");
            
            //Outputs all accuracy metrics and confusion matrix
            //System.out.println(eval.toSummaryString());
            //System.out.println(eval.toMatrixString());          
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
    
    /**
     * Evaluates the classifier using 10-fold cross validation.
     * 
     * @param cl The classifier
     */
    public void evaluateCV(Classifier cl)
    {
        try
        {
            cl.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cl, data, 10, new java.util.Random(1));
            System.out.println("Accuracy: " + df.format(eval.correct()/data.numInstances()*100.0) + "%");            
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
