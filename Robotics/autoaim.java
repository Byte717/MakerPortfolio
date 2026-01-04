package frc.robot.util;
import edu.wpi.first.math.geometry.Pose2d;
import frc.robot.Constants;
import frc.robot.subsystems.drive.Drive;
import edu.wpi.first.math.geometry.Translation3d;
import edu.wpi.first.wpilibj.DriverStation;

public class Projectile {
    /*
     * Change teh verlocity, deceleration, and height of the target
     */
    private double precision;
    private double v; // speed of the shooter, 35 mph, 15 m/s
    private double a; // deceleration in m/s^2
    private double height; // placeholder, height from shooter to target, tbd
    private double d;
    private double fixedTheta;
    private int MAX_ITERATIONS;
    public final Translation3d target;
    public final Drive drive;
   
    public Projectile(Drive d){
        if (DriverStation.getAlliance().get() == DriverStation.Alliance.Red){
            target = Constants.AutoAimConstants.targetRed
        } else target = Constants.AutoAimConstants.targetBlue;

        precision = Constants.AutoAimConstants.precision;
        v = Constants.AutoAimConstants.v;
        a = Constants.AutoAimConstants.a;
        height = target.getZ();
        MAX_ITERATIONS = Constants.AutoAimConstants.MAX_ITERATIONS;
        drive = d;
    }

    private double getDistance(){ // get distance from target
        return Math.sqrt((Math.pow((drive.getPose().getX() - target.getX()),2)+Math.pow((drive.getPose().getY() - target.getY()),2)));
    }

    private boolean testTheta(double theta){
        double sin = Math.sin(theta), cos = Math.cos(theta);
        double t = (-v*sin + Math.sqrt((v*v)*(cos*cos) + 2*a*d))/a;
        double h = v*sin*t - 4.9*t*t;
        return h > height;
    }

    private double optimalAngle(){ // finds optimal shooter angle to shoot at target
       d = this.getDistance();
       double low = Constants.AutoAimConstants.lowAngle, high = Constants.AutoAimConstants.highAngle, mid = 0.0;
       int i = 0;
       while(high - low > precision && i < MAX_ITERATIONS){
            i++;
            mid = (high + low)/2;
            if(this.testTheta(mid)){
                high = mid;
            }else{
                low = mid;
            }
       }
       return (high+low)/2;
    }

    private double optimalDistance(){ // returns how far from the target the robot should be to make it in
        double sin = Math.sin(fixedTheta), cos = Math.cos(fixedTheta);
        double t = (v*sin - Math.sqrt((v*v)*(sin*sin) - 4*4.9*height))/9.8;
        return v*sin + 0.5*a*t*t;
    }

    public Command aim(){
        double theta = this.optimalAngle();
        // command to rotate to heading, change angle of the shooter, and then shoot
        return Commands.runOnce(); // work in progress here
    }

    public Command aimFixed(){ // aims at the target by moving closer in case of a fixed angle
        double dist = this.optimalDistance();
        return Commands.runOnce();
    }
}

