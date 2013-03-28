/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    InfoGainAttributeEval.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.attributeSelection;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToBinary;
import org.apache.commons.math3.distribution.NormalDistribution;

/** 
 <!-- globalinfo-start -->
 * BiNormalSeperationEval:<br/>
 * <br/>
 * Evaluates the worth of an attribute by measuring the F_1(tpr) - F_1(fpr) where F_1 is the
 * z-score with respect to the class.<br/>
 * <br/>
 * InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).<br/>
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -M
 *  treat missing values as a seperate value.</pre>
 * 
 * <pre> -B
 *  just binarize numeric attributes instead 
 *  of properly discretizing them.</pre>
 * 
 <!-- options-end -->
 *
 * @author Anthony Rios (anthonymrios@gmail.com)
 * @version $Revision: 8034 $
 * @see Discretize
 * @see NumericToBinary
 */
public class BiNormalSeperationREval
  extends ASEvaluation
  implements AttributeEvaluator, OptionHandler {
  
  /** The z-score for each attribute */
  private double[] m_zScores;

  /**
   * Returns a string describing this attribute evaluator
   * @return a description of the evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "BiNormSeperationEval:\n\nEvaluates the worth of an attribute "
      +"by taking F_1(tpr) - F_1(fpr) F_1 is the z-score and fpr and tpr\n\n"+
      " represent the true and false positive rate with respect to the class.\n";
  }

  /**
   * Constructor
   */
  public BiNormalSeperationREval() {
    System.out.println("TEST3");
    resetOptions();
    System.out.println("TEST2");
  }

/**
   * Gets the current settings of WrapperSubsetEval.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions () {
    String[] options = new String[0];
    return options;
  }

 /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions (String[] options)
    throws Exception {
  }

  /**
   * Returns an enumeration describing the available options.
   * @return an enumeration of all the available options.
   **/
  public Enumeration listOptions () {
    Vector newVector = new Vector(2);
    return  newVector.elements();
  }


  /**
   * Returns the capabilities of this evaluator.
   *
   * @return            the capabilities of this evaluator
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();
    
    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);
    
    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    
    return result;
  }

  /**
   * Initializes an BNS attribute evaluator.
   * Discretizes all attributes that are numeric.
   *
   * @param data set of instances serving as training data 
   * @throws Exception if the evaluator has not been 
   * generated successfully
   */
  public void buildEvaluator (Instances data)
    throws Exception {
    
    // can evaluator handle data?
    getCapabilities().testWithFail(data);
    System.out.println("TEST1");

    int classIndex = data.classIndex();
    int numInstances = data.numInstances();
    
    int numClasses = data.attribute(classIndex).numValues();

    double[] tp = new double[data.numAttributes()];
    double[] fp = new double[data.numAttributes()];
    double[] totalPos = new double[data.numAttributes()];
    double[] totalNeg = new double[data.numAttributes()];
    // Initialize values
    for(int i = 0; i < data.numAttributes(); i++) {
      tp[i] = 0;
      fp[i] = 0;
      totalPos[i] = 0;
      totalNeg[i] = 0;
    }
    System.out.println("TEST0");

    Instance curInst;
    String classValue;
    double attValue;
    for(int i = 0; i < numInstances; i++)  {
      curInst = data.get(i);
      classValue = curInst.stringValue(classIndex);
      for(int j = 0; j < data.numAttributes(); j++) {
        if(j != classIndex) {
          attValue = curInst.value(j);
          if(classValue.equals("1")) totalPos[j]++;
          if(classValue.equals("0")) totalNeg[j]++;
          if(classValue.equals("1") && attValue > 0) tp[j]++;
          if(classValue.equals("0") && attValue == 0) fp[j]++;
        }
      }
    }
    System.out.println("TEST");

    double[] tpr = new double[data.numAttributes()];
    double[] fpr = new double[data.numAttributes()];
    NormalDistribution nd = new NormalDistribution();
    m_zScores = new double[data.numAttributes()];
    for(int i = 0; i < data.numAttributes(); i++) {
      tpr[i] = tp[i]/totalPos[i];
      fpr[i] = fp[i]/totalNeg[i];
      if(tp[i] == 0) tpr[i] = 0.00005;
      if(fp[i] == 0) fpr[i] = 0.00005;
      m_zScores[i] = nd.inverseCumulativeProbability(fpr[i]) - nd.inverseCumulativeProbability(tpr[i]);
    }

  }

  /**
   * Reset options to their default values
   */
  protected void resetOptions () {
  }


  /**
   * evaluates an individual attribute by measuring the amount
   * of information gained about the class given the attribute.
   *
   * @param attribute the index of the attribute to be evaluated
   * @return the info gain
   * @throws Exception if the attribute could not be evaluated
   */
  public double evaluateAttribute (int attribute)
    throws Exception {

    return m_zScores[attribute];
  }

  /**
   * Describe the attribute evaluator
   * @return a description of the attribute evaluator as a string
   */
  public String toString () {
    StringBuffer text = new StringBuffer();

    if (m_zScores == null) {
      text.append("Information Gain attribute evaluator has not been built");
    }
    
    text.append("\n");
    return  text.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 8034 $");
  }
  
  // ============
  // Test method.
  // ============
  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main (String[] args) {
    runEvaluator(new InfoGainAttributeEval(), args);
  }
}
