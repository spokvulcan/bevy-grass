//! `GrassSettings::strand_lods`: the blades of the far levels are written as
//! strand records and drawn as no triangle.

use super::*;
use crate::api::{GrassVariation, GroundTint};
use crate::fields::FieldTables;
use crate::mirror::{STRAND_ALONG, srgb_word, strand_colours, strand_tip, unpack_bend};
use crate::render::view::{GrassStrands, STRAND_SIZE};

/// One record read back: foot, tall, beside, root and tip colour.
#[derive(Debug, Clone, Copy)]
struct Record {
    foot: Vec3,
    tall: f32,
    beside: Vec2,
    root: u32,
    tip: u32,
}

/// The records of the first grass view, and its `GrassStrands` if it has one.
fn gpu_strands(app: &mut App) -> (Vec<Record>, Option<GrassStrands>) {
    let world = app.sub_app_mut(RenderApp).world_mut();
    let device = world.resource::<RenderDevice>().clone();
    let queue = world.resource::<RenderQueue>().clone();
    let mut views = world.query::<(&GrassViewBuffers, Option<&GrassStrands>)>();
    let (buffers, handed) = views.iter(world).next().expect("a grass view");
    let handed = handed.cloned();
    let (records, count) = buffers.strand_buffers();
    let count = read_buffer(&device, &queue, count, 4)[0];
    let words = read_buffer(&device, &queue, records, records.size());
    let f = |i: usize| f32::from_bits(words[i]);
    let out = (0..count as usize)
        .map(|n| {
            let i = n * (STRAND_SIZE as usize / 4);
            Record {
                foot: Vec3::new(f(i), f(i + 1), f(i + 2)),
                tall: f(i + 3),
                beside: Vec2::new(f(i + 4), f(i + 5)),
                root: words[i + 6],
                tip: words[i + 7],
            }
        })
        .collect();
    (out, handed)
}

/// The record `generate` should write for the blade of `words`: the same
/// sums on the processor, for a field with none of the optional look terms:
/// its colours fitted to a line as `mirror::strand_colours` fits them.
fn expected(
    words: [u32; 4],
    view: &ViewUniformData,
    fields: &[crate::gpu_layout::FieldGpu],
) -> Record {
    let origin = view.instance_origin;
    let position =
        |q: u32, centre: f32, range: f32| ((q as f32 / 65535.0) - 0.5) * 2.0 * range + centre;
    let foot = Vec3::new(
        position(words[0] & 0xffff, origin.x, origin.w),
        position(words[1] & 0xffff, origin.y, view.limits.w),
        position(words[0] >> 16, origin.z, origin.w),
    );
    let facing = ((words[1] >> 16) & 0xff) as f32 / 256.0;
    let seed = (words[1] >> 24) as f32 / 255.0;
    let clump = (words[2] >> 26) as f32 / 63.0;
    let field = &fields[((words[2] >> 16) & 0xff) as usize];
    let height = (words[2] & 0xff) as f32 / 255.0 * field.shape.y;
    let bend = Vec2::new(
        unpack_bend(words[3] & 0xffff).0,
        unpack_bend(words[3] >> 16).0,
    );

    let eye = view.camera_position.truncate();
    let to_eye = (Vec2::new(eye.x, eye.z) - Vec2::new(foot.x, foot.z)).normalize();
    let yaw = (facing + field.anchor.z) * std::f32::consts::TAU;
    let face = Vec2::new(yaw.sin(), yaw.cos());
    let facing_eye = face.dot(to_eye);
    let edge_on = 1.0 - facing_eye.abs();
    let toward = to_eye * if facing_eye >= 0.0 { 1.0 } else { -1.0 };
    let face = face.lerp(toward, edge_on * view.limits.z).normalize();
    let tip = strand_tip(height, face, field.motion.x, bend, 0.0);

    let colour = |t: f32| {
        let shade = (0.82 + (1.12 - 0.82) * clump) * (0.94 + 0.12 * seed);
        let green = field
            .root_color
            .truncate()
            .lerp(field.tip_color.truncate(), t.powf(0.7))
            * shade;
        let straw = Vec3::splat(green.dot(Vec3::new(0.3, 0.59, 0.11))) * Vec3::new(1.25, 1.1, 0.6);
        let dry = if (clump * 5.3).fract() >= 0.8 {
            0.4
        } else {
            0.0
        };
        let shaded = field.motion.w * (1.0 - smoothstep(0.0, 0.6, t));
        green.lerp(straw, dry) * (1.0 - shaded)
    };
    let (root, tip_colour) = strand_colours(STRAND_ALONG.map(colour));
    Record {
        foot,
        tall: tip.length(),
        beside: Vec2::new(tip.x, tip.z),
        root: srgb_word(root),
        tip: srgb_word(tip_colour),
    }
}

fn smoothstep(low: f32, high: f32, x: f32) -> f32 {
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Whether two colour words differ by at most one step in each channel.
fn near(a: u32, b: u32) -> bool {
    (0..3).all(|c| ((a >> (8 * c)) & 0xff).abs_diff((b >> (8 * c)) & 0xff) <= 1)
}

/// The blades of the middle and the coarse level become strands, one record
/// each with the blade's root, tip and colours as the draw has them, and no
/// instance; the near blades stay as they were. And with the setting as it
/// comes, nothing is handed out.
#[test]
#[ignore = "needs a GPU: cargo test -- --ignored gpu"]
fn gpu_far_blades_are_written_as_strands_and_drawn_as_none() {
    let mut app = headless_app();
    // A style with none of the optional look terms, whose colours the sums
    // above follow.
    {
        let world = app.world_mut();
        let mut styles = world.resource_mut::<Assets<GrassStyle>>();
        let id = styles.ids().next().expect("the meadow");
        let mut style = styles.get_mut(id).expect("the meadow");
        style.ground_tint = GroundTint {
            amount: 0.0,
            ..style.ground_tint
        };
        style.variation = GrassVariation {
            patch_amount: 0.0,
            dry_tips: 0.0,
            dead_share: 0.0,
            ..style.variation
        };
        style.blade.twist = 0.0;
        style.blade.droop = 0.0;
        style.blade.fold = 0.0;
        style.blade.head_length = 0.0;
    }
    wait_for_grass(&mut app);
    // The clock stands still, so the wind bends each blade the same in both
    // halves of the test.
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    for _ in 0..3 {
        app.update();
    }
    let (before_view, before) = gpu_words_and_lods(&mut app).expect("grass");
    let (none, handed) = gpu_strands(&mut app);
    assert!(
        none.is_empty() && handed.is_none(),
        "no strands as the setting comes"
    );
    let far: Vec<[u32; 4]> = before
        .iter()
        .filter(|(_, lod)| *lod >= 1)
        .map(|(words, _)| *words)
        .collect();
    let near_blades = before.len() - far.len();
    assert!(
        far.len() > 1_000 && near_blades > 1_000,
        "{} far, {near_blades} near",
        far.len()
    );

    app.insert_resource(crate::GrassSettings {
        strand_lods: 2,
        ..crate::GrassSettings::HIGH
    });
    for _ in 0..5 {
        app.update();
    }
    let (view, after) = gpu_words_and_lods(&mut app).expect("grass");
    let (strands, handed) = gpu_strands(&mut app);
    let handed = handed.expect("the strands are handed out");
    assert_eq!(view.strand_lods, 2);
    assert_eq!(handed.capacity, view.strand_capacity);
    assert_eq!(
        view.instance_origin, before_view.instance_origin,
        "the same eye"
    );
    // No triangle for a far blade, and every near blade as before.
    assert!(
        after.iter().all(|(_, lod)| *lod == 0),
        "a far blade is still an instance"
    );
    assert_eq!(after.len(), near_blades);
    // A strand for every far blade the camera sees: the test's camera casts
    // no shadow of its own, so every far blade is one it sees.
    assert_eq!(strands.len(), far.len());

    let fields = app.world().resource::<FieldTables>().fields.clone();
    let mut wanted: Vec<Record> = far.iter().map(|&w| expected(w, &view, &fields)).collect();
    let mut got = strands.clone();
    got.sort_by(|a, b| {
        a.foot
            .x
            .total_cmp(&b.foot.x)
            .then(a.foot.z.total_cmp(&b.foot.z))
    });
    wanted.sort_by(|a, b| {
        a.foot
            .x
            .total_cmp(&b.foot.x)
            .then(a.foot.z.total_cmp(&b.foot.z))
    });
    let mut colours_off = 0;
    for (g, w) in got.iter().zip(&wanted) {
        assert!(
            (g.foot - w.foot).length() < 1e-3,
            "{g:?} is not the blade {w:?}"
        );
        assert!((g.tall - w.tall).abs() < 1e-4, "{g:?} against {w:?}");
        assert!((g.beside - w.beside).length() < 1e-4, "{g:?} against {w:?}");
        assert!(g.beside.length() <= g.tall + 1e-6);
        assert_eq!((g.root | g.tip) >> 24, 0, "the top byte is spare");
        if !near(g.root, w.root) || !near(g.tip, w.tip) {
            colours_off += 1;
        }
    }
    assert_eq!(colours_off, 0, "of {} strands", got.len());

    // Back to none: nothing handed out, and the far blades are triangles again.
    app.insert_resource(crate::GrassSettings::HIGH);
    for _ in 0..5 {
        app.update();
    }
    let (_, again) = gpu_words(&mut app).expect("grass");
    let (strands, handed) = gpu_strands(&mut app);
    assert!(strands.is_empty() && handed.is_none());
    assert_eq!(again.len(), before.len());
}
