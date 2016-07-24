
public class Main {

	final static int NUM_FEATURES=2;
	final static int NUM_EXAMPLES=4;
	final static double[][] INPUTS = {{1,2},{2,4},{3,6},{1,1}};
	final static double[] OUTPUTS = {3,6,9,2};
	final static double[][] NORMALIZED_INPUTS =(INPUTS);
	final static double LEARNING_RATE = 0.00001;
	final static int NUM_ITERATIONS = 100000000;
	
	
	
	public static void main(String[] args) {
		double[] coefficients = getCoefficients();
		for (int i=0;i<coefficients.length;i++){
			System.out.println(coefficients[i]);
		}
		
	}
	
	public static double partialDerivative(int featureIndex, double[] coefficients){
		double derivative = 0;
		if (featureIndex == -1){
			for (int i=0;i<NUM_EXAMPLES;i++){
				derivative+=((1.0/NUM_EXAMPLES)*(hypothesis (i, coefficients)-OUTPUTS[i]));
			}
			return derivative;
		}
		else{
			for (int i=0;i<NUM_EXAMPLES;i++){
				derivative+=(1.0/NUM_EXAMPLES)*NORMALIZED_INPUTS[i][featureIndex]*(hypothesis (i, coefficients)-OUTPUTS[i]);
			}
			return derivative;
		}
	}
	
	public static double error (double[] coefficients){
		double error = 0;
		for (int i=0;i<NUM_EXAMPLES;i++){
			error+=Math.pow((hypothesis (i, coefficients)-OUTPUTS[i]),2);
		}
		return error/(2.0*NUM_EXAMPLES);
	}
	
	public static double[] getCoefficients (){
		//Initialize the coefficients as 0s
		double[] coefficients = {0,0,0};
		double[] origCoefficients = coefficients;
		//Batch gradient descent
		for (int i=0;i<NUM_ITERATIONS;i++){
			for (int j=0;j<3;j++){
				coefficients[j]=origCoefficients[j]-LEARNING_RATE*partialDerivative(j-1, origCoefficients);
				
			}
			
			//System.out.println(error(coefficients));
			origCoefficients = coefficients;
		}
		return coefficients; 
	}
	
	
	public static double hypothesis (int exampleIndex, double[] coefficients){
		double hypothesis = 0;
		hypothesis+=coefficients[0];
		for (int i=1;i< NUM_FEATURES+1; i++){
			hypothesis += coefficients[i]*NORMALIZED_INPUTS[exampleIndex][i-1];
		}
		return hypothesis;
	}
	
	public static double[][] normalize (double[][] matrix){
		double[] max = new double[NUM_FEATURES];
		double[] min = new double[NUM_FEATURES];
		double[] averages = new double[NUM_FEATURES];
		double[][] normalizedMatrix = new double[NUM_EXAMPLES][NUM_FEATURES];
		for (int j=0;j<NUM_FEATURES;j++){
			max[j]=matrix[0][j];
			min[j]=matrix[0][j];
			for (int i=0;i<NUM_EXAMPLES;i++){
				max[j]=Math.max(max[j], matrix[i][j]);
				min[j]=Math.min(min[j], matrix[i][j]);
				averages[j]+=matrix[i][j];
			}
		}
		for (int i=0;i<NUM_FEATURES;i++){
			averages[i]=averages[i]/NUM_EXAMPLES;
		}
		
		for (int i=0;i<NUM_FEATURES;i++){
			for (int j=0;j<NUM_EXAMPLES;j++){
				normalizedMatrix[j][i]=(matrix[j][i]-averages[i])/(max[i]-min[i]);
			}
		}
		
		return normalizedMatrix;
		
		
	
	}

}
