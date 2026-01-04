public class BellCurveProfile{
    private double halfTime;     // a
    private double deceleration; // b
    private double maxVelocity;  // c
    private double positionCoef;
    private double positionOffset;
   
    public BellCurveProfile(double a, double b, double c){
        this.halfTime = a;
        this.deceleration = b;
        this.maxVelocity = c;
        updateEquationParams();
    }
    public void updateEquationParams(){
        this.positionCoef = this.maxVelocity*Math.sqrt(Math.PI*this.deceleration)*0.5;
        this.positionOffset = this.positionCoef*PathVelocity.erf(this.halfTime/Math.sqrt(this.deceleration));
    }
    public double getVelocity(double time){
        double t = (time - this.halfTime);
        return this.maxVelocity * Math.exp(-(t*t)/this.deceleration);
    }
    public double getPosition(double time){
        double t = (time - this.halfTime) / Math.sqrt(this.deceleration);
        return this.positionCoef * erf(t) + this.positionOffset;
    }
    public double getAcceleration(double time){
        return -getVelocity(time)*(2*(time - this.halfTime)/this.deceleration);
    }
    public double getMaxAcceleration(){
        double t = a - Math.sqrt(this.deceleration/2);
        return getAcceleration(t);
    }
    public static double erf(double z) {
        int sign = (x >= 0) ? 1 : -1;
        x = Math.abs(x);
        // constants
        double a1 =  0.254829592;
        double a2 = -0.284496736;
        double a3 =  1.421413741;
        double a4 = -1.453152027;
        double a5 =  1.061405429;
        double p  =  0.3275911;
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
        return sign * y; // erf(-x) = -erf(x)
    }
    public static double[] getParams(double maxVel, double targetDist){
        double elipson = Constants.BellConstants.elipson;
        double rootPi = Math.sqrt(Math.PI);
        double denom = rootPi*maxVel*(1+erf(elipson/maxVel));
        double numer = 2*targetDist;
        double b = Math.pow((numer/denom),2);
        double a = Math.sqrt(-b*Math.log(elipson/maxVel));
        return new double[]{a,b};
    }
}