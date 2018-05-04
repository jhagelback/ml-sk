
package aml;

import java.text.DecimalFormat;
import weka.core.*;
import weka.core.converters.ConverterUtils;
import weka.classifiers.lazy.IBk;
import weka.classifiers.*;

/**
 * Regression using Weka.
 * 
 * @author Johan Hagelbäck (johan.hagelback@lnu.se)
 */
public class Regression 
{
    /** Path to dataset */
    private String filename;
    /** Dataset in Weka instances */
    private Instances data;
    /** Weka classifier */
    private Classifier cl;
    
    private DecimalFormat df = new DecimalFormat("0.00"); 
    
    /**
     * Runs everything
     */
    public static void run()
    {
        Regression r = new Regression("data/GPUbenchmark.arff");
        
        System.out.println("Test regression:");
        r.testRegression();
    }
    
    /**
     * Constructor.
     * 
     * @param filename Path to dataset file
     */
    public Regression(String filename)
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
     * Test regression accuracy by training on 18 instances and testing on
     * 1 instance, and calculate average difference for all instances.
     */
    public void testRegression()
    {
        try
        {
            double avg_diff = 0.0;
            double sq_diff = 0.0;
            double avg_diff_perc = 0.0;
            
            for (int i = 0; i < 19; i++)
            {
                //Read data
                readData();
                
                //Optional: remove unnecessary attributes
                //data.deleteAttributeAt(1);
                
                //Remove instance
                Instance inst = data.remove(i);

                cl = new IBk(3);
                cl.buildClassifier(data);
                
                //Actual and predicted benchmark values
                double estimated = cl.classifyInstance(inst);
                double actual = inst.classValue();
                
                //Diffs
                double diff = Math.abs(estimated - actual);
                sq_diff += Math.pow(estimated - actual, 2);
                avg_diff += diff;
                double diff_perc = diff / actual * 100.0;
                avg_diff_perc += diff_perc;
                
                //Output
                System.out.println("Predicted: " + df.format(estimated) + " (actual " + actual + ") -> Diff " + df.format(diff) + " (" + df.format(diff_perc) + "%)");
            }
            avg_diff /= 19;
            sq_diff = Math.sqrt(sq_diff / 19);
            avg_diff_perc /= 19;
            System.out.println("Average diff: " + df.format(avg_diff) + " Squared diff: " + df.format(sq_diff) + " (" + df.format(avg_diff_perc) + "%)");
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
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
            cl = new IBk(3);
            cl.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(cl, data);
            System.out.println(eval.toSummaryString());
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
            cl = new IBk(3);
            cl.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cl, data, 10, new java.util.Random(1));
            System.out.println(eval.toSummaryString());
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
